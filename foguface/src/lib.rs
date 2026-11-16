//! foguface — detecção, reconhecimento e identificação de faces com
//! [insight-face-rs](https://crates.io/crates/insight-face-rs) e API C para o `fogu:ffi`.
//!
//! Convenções (Fogu 0.7+; callbacks em `cb.rs`):
//! - Com um sink registrado (`fface_set_sink`), `buf = NULL, cap = 0` entrega a
//!   saída inteira pelo callback: a inferência nunca roda duas vezes por causa
//!   de um buffer pequeno.
//! - Motores e galerias são identificados por handles `i64` (> 0).
//! - Funções de status retornam `0` em sucesso e `-1` em erro; a mensagem
//!   fica disponível em `fface_last_error`.
//! - Saídas são escritas num buffer do chamador (`ffi.alloc`), com NUL no fim
//!   quando são texto. O retorno é o tamanho escrito, ou `-(tamanho necessário)`
//!   se o buffer for pequeno demais.
//! - Números reais (scores, similaridades, embeddings) trafegam em JSON.

mod cb;
mod draw;
mod gallery;

use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use image::RgbImage;
use insight_face_rs::{Face, FaceEmbedding, FaceEngine, FaceEngineConfig};
use serde_json::{Value, json};

pub use gallery::{Gallery, Match};

static ENGINES: LazyLock<Mutex<HashMap<i64, Arc<FaceEngine>>>> = LazyLock::new(Default::default);
static GALLERIES: LazyLock<Mutex<HashMap<i64, Arc<Mutex<Gallery>>>>> = LazyLock::new(Default::default);
static NEXT_HANDLE: AtomicI64 = AtomicI64::new(1);
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

fn set_error(msg: impl Into<String>) {
    *LAST_ERROR.lock().unwrap() = msg.into();
}

/// Executa `f` sem deixar um panic atravessar a fronteira C.
fn guard<T>(on_error: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(e);
            on_error
        }
        Err(_) => {
            set_error("internal panic in foguface");
            on_error
        }
    }
}

fn status(f: impl FnOnce() -> Result<(), String>) -> i32 {
    guard(-1, || f().map(|_| 0))
}

fn engine(handle: i64) -> Result<Arc<FaceEngine>, String> {
    ENGINES.lock().unwrap().get(&handle).cloned().ok_or_else(|| format!("invalid engine handle {handle}"))
}

fn gallery(handle: i64) -> Result<Arc<Mutex<Gallery>>, String> {
    GALLERIES.lock().unwrap().get(&handle).cloned().ok_or_else(|| format!("invalid gallery handle {handle}"))
}

fn new_handle() -> i64 {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

unsafe fn text<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Ok("");
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|_| "string is not valid UTF-8".to_string())
}

unsafe fn write_out(buf: *mut u8, cap: i64, s: &str) -> i64 {
    if let Some(n) = cb::try_sink(buf, cap, s.as_bytes()) {
        return n;
    }
    let need = s.len() + 1;
    if buf.is_null() || cap < need as i64 {
        return -(need as i64);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), buf, s.len());
        *buf.add(s.len()) = 0;
    }
    s.len() as i64
}

/// Opções de `fface_engine_new`.
struct EngineOptions {
    det_model: String,
    rec_model: String,
    idle_timeout: Duration,
}

impl EngineOptions {
    fn from_json(s: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(s).map_err(|e| format!("invalid options JSON: {e}"))?;
        let path = |key: &str, default: &str| v.get(key).and_then(Value::as_str).unwrap_or(default).to_string();
        Ok(Self {
            det_model: path("det_model", "models/det_10g.onnx"),
            rec_model: path("rec_model", "models/w600k_r50.onnx"),
            idle_timeout: Duration::from_secs(v.get("idle_timeout").and_then(Value::as_u64).unwrap_or(0)),
        })
    }
}

fn load_image(path: &str) -> Result<RgbImage, String> {
    Ok(image::open(path).map_err(|e| format!("{path}: {e}"))?.to_rgb8())
}

/// Decodifica uma imagem (JPEG, PNG, WebP...) que está na memória.
fn decode_image(data: *const u8, len: i64) -> Result<RgbImage, String> {
    if data.is_null() || len <= 0 {
        return Err("empty image buffer".into());
    }
    let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) };
    Ok(image::load_from_memory(bytes).map_err(|e| format!("decode image: {e}"))?.to_rgb8())
}

/// Rosto em JSON, com coordenadas em pixels. `with_embedding` inclui o vetor de 512 posições.
fn face_json(face: &Face, width: u32, height: u32, with_embedding: bool) -> Value {
    let b = face.bbox.to_absolute(width, height);
    let lm = face.landmarks.to_absolute(width, height);
    let mut v = json!({
        "score": face.score,
        "bbox": { "x": b.x1, "y": b.y1, "width": b.x2 - b.x1, "height": b.y2 - b.y1 },
        "landmarks": lm.0,
    });
    if with_embedding {
        v["embedding"] = json!(face.embedding.0.to_vec());
    }
    v
}

/// Maior rosto da imagem (o mais provável de ser "a pessoa" num retrato).
fn largest(faces: Vec<Face>) -> Result<Face, String> {
    faces
        .into_iter()
        .max_by(|a, b| a.bbox.area().total_cmp(&b.bbox.area()))
        .ok_or_else(|| "no face found in image".to_string())
}

fn run(handle: i64, img: &RgbImage) -> Result<Vec<Face>, String> {
    engine(handle)?.run(img).map_err(|e| e.to_string())
}

fn detect_json(handle: i64, img: &RgbImage, with_embedding: bool) -> Result<String, String> {
    let faces = run(handle, img)?;
    let list: Vec<Value> = faces.iter().map(|f| face_json(f, img.width(), img.height(), with_embedding)).collect();
    Ok(json!({ "width": img.width(), "height": img.height(), "faces": list }).to_string())
}

pub(crate) fn embedding_from_json(v: &Value) -> Result<FaceEmbedding, String> {
    let arr = v.as_array().ok_or("embedding must be an array of 512 numbers")?;
    if arr.len() != 512 {
        return Err(format!("embedding must have 512 numbers, got {}", arr.len()));
    }
    let mut out = [0f32; 512];
    for (o, x) in out.iter_mut().zip(arr) {
        *o = x.as_f64().ok_or("embedding must contain only numbers")? as f32;
    }
    Ok(FaceEmbedding(out))
}

/// Versão da biblioteca. Retorna o tamanho escrito (ver convenções).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_version(buf: *mut u8, cap: i64) -> i64 {
    unsafe { write_out(buf, cap, env!("CARGO_PKG_VERSION")) }
}

/// Última mensagem de erro. Retorna o tamanho escrito (0 se não há erro).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_last_error(buf: *mut u8, cap: i64) -> i64 {
    let msg = LAST_ERROR.lock().unwrap().clone();
    unsafe { write_out(buf, cap, &msg) }
}

// ---------------------------------------------------------------------------
// Motor: detecção + embeddings
// ---------------------------------------------------------------------------

/// Carrega os modelos ONNX. `options_json`: `det_model`, `rec_model`, `idle_timeout`
/// (segundos sem uso até liberar a memória dos modelos; 0 desliga). Retorna o handle ou -1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_engine_new(options_json: *const c_char) -> i64 {
    guard(-1, || {
        let s = unsafe { text(options_json) }?;
        let opts = EngineOptions::from_json(if s.is_empty() { "{}" } else { s })?;
        for p in [&opts.det_model, &opts.rec_model] {
            if !std::path::Path::new(p).is_file() {
                return Err(format!("model not found: {p}"));
            }
        }
        let config = FaceEngineConfig::new(&opts.det_model, &opts.rec_model, opts.idle_timeout);
        let engine = Arc::new(FaceEngine::new(&config).map_err(|e| e.to_string())?);
        if !opts.idle_timeout.is_zero() {
            FaceEngine::start_reaper_thread(&engine);
        }
        let handle = new_handle();
        ENGINES.lock().unwrap().insert(handle, engine);
        Ok(handle)
    })
}

/// Libera o motor (e os modelos carregados).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_engine_free(handle: i64) -> i32 {
    status(|| ENGINES.lock().unwrap().remove(&handle).map(|_| ()).ok_or_else(|| format!("invalid engine handle {handle}")))
}

/// Detecta os rostos de um arquivo de imagem. Escreve JSON
/// `{width, height, faces: [{score, bbox: {x, y, width, height}, landmarks: [[x, y] x5], embedding?}]}`.
/// `with_embedding` != 0 inclui os vetores de 512 posições.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_detect_file(
    handle: i64,
    path: *const c_char,
    with_embedding: i32,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    guard(-1, || {
        let img = load_image(unsafe { text(path) }?)?;
        Ok(unsafe { write_out(buf, cap, &detect_json(handle, &img, with_embedding != 0)?) })
    })
}

/// Igual a `fface_detect_file`, para uma imagem codificada (JPEG, PNG...) na memória.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_detect_bytes(
    handle: i64,
    data: *const u8,
    len: i64,
    with_embedding: i32,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    guard(-1, || {
        let img = decode_image(data, len)?;
        Ok(unsafe { write_out(buf, cap, &detect_json(handle, &img, with_embedding != 0)?) })
    })
}

/// Compara o maior rosto de cada imagem. Escreve JSON `{similarity, same, threshold}`.
/// `threshold_milli` é o limiar em milésimos (ex.: 400 = 0.40; <= 0 usa o padrão).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_compare_files(
    handle: i64,
    path_a: *const c_char,
    path_b: *const c_char,
    threshold_milli: i64,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    guard(-1, || {
        let a = largest(run(handle, &load_image(unsafe { text(path_a) }?)?)?)?;
        let b = largest(run(handle, &load_image(unsafe { text(path_b) }?)?)?)?;
        let threshold = gallery::threshold(threshold_milli);
        let sim = a.embedding.cosine_similarity(&b.embedding);
        let out = json!({ "similarity": sim, "same": sim >= threshold, "threshold": gallery::threshold_json(threshold) });
        Ok(unsafe { write_out(buf, cap, &out.to_string()) })
    })
}

/// Similaridade de cosseno entre dois embeddings em JSON (`[...512]` cada). Escreve `{similarity}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_similarity(a_json: *const c_char, b_json: *const c_char, buf: *mut u8, cap: i64) -> i64 {
    guard(-1, || {
        let parse = |p| -> Result<FaceEmbedding, String> {
            let v: Value = serde_json::from_str(unsafe { text(p) }?).map_err(|e| format!("invalid embedding JSON: {e}"))?;
            embedding_from_json(&v)
        };
        let sim = parse(a_json)?.cosine_similarity(&parse(b_json)?);
        Ok(unsafe { write_out(buf, cap, &json!({ "similarity": sim }).to_string()) })
    })
}

// ---------------------------------------------------------------------------
// Galeria: cadastro de pessoas e identificação (1:N)
// ---------------------------------------------------------------------------

/// Cria uma galeria vazia. Retorna o handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_new() -> i64 {
    guard(-1, || {
        let handle = new_handle();
        GALLERIES.lock().unwrap().insert(handle, Arc::new(Mutex::new(Gallery::default())));
        Ok(handle)
    })
}

/// Carrega uma galeria salva com `fface_gallery_save`. Retorna o handle ou -1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_load(path: *const c_char) -> i64 {
    guard(-1, || {
        let g = Gallery::load(unsafe { text(path) }?)?;
        let handle = new_handle();
        GALLERIES.lock().unwrap().insert(handle, Arc::new(Mutex::new(g)));
        Ok(handle)
    })
}

/// Salva a galeria num arquivo JSON (nomes + embeddings).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_save(handle: i64, path: *const c_char) -> i32 {
    status(|| gallery(handle)?.lock().unwrap().save(unsafe { text(path) }?))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_free(handle: i64) -> i32 {
    status(|| GALLERIES.lock().unwrap().remove(&handle).map(|_| ()).ok_or_else(|| format!("invalid gallery handle {handle}")))
}

/// Cadastra o maior rosto de uma imagem com o nome `name`. Uma pessoa pode ter
/// várias amostras (chame de novo com outras fotos). Retorna o nº de amostras dela ou -1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_enroll_file(gallery_h: i64, engine_h: i64, name: *const c_char, path: *const c_char) -> i64 {
    guard(-1, || {
        let name = unsafe { text(name) }?;
        let face = largest(run(engine_h, &load_image(unsafe { text(path) }?)?)?)?;
        Ok(gallery(gallery_h)?.lock().unwrap().enroll(name, face.embedding) as i64)
    })
}

/// Igual a `fface_gallery_enroll_file`, para uma imagem codificada na memória.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_enroll_bytes(
    gallery_h: i64,
    engine_h: i64,
    name: *const c_char,
    data: *const u8,
    len: i64,
) -> i64 {
    guard(-1, || {
        let name = unsafe { text(name) }?;
        let face = largest(run(engine_h, &decode_image(data, len)?)?)?;
        Ok(gallery(gallery_h)?.lock().unwrap().enroll(name, face.embedding) as i64)
    })
}

/// Cadastra um embedding já calculado (JSON `[...512]`). Retorna o nº de amostras da pessoa.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_enroll_embedding(handle: i64, name: *const c_char, embedding_json: *const c_char) -> i64 {
    guard(-1, || {
        let name = unsafe { text(name) }?;
        let v: Value = serde_json::from_str(unsafe { text(embedding_json) }?).map_err(|e| format!("invalid embedding JSON: {e}"))?;
        Ok(gallery(handle)?.lock().unwrap().enroll(name, embedding_from_json(&v)?) as i64)
    })
}

/// Remove uma pessoa (todas as amostras). Retorna 1 se existia, 0 se não.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_remove(handle: i64, name: *const c_char) -> i64 {
    guard(-1, || Ok(gallery(handle)?.lock().unwrap().remove(unsafe { text(name) }?) as i64))
}

/// Lista as pessoas: JSON `[{name, samples}]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_gallery_list(handle: i64, buf: *mut u8, cap: i64) -> i64 {
    guard(-1, || {
        let list = gallery(handle)?.lock().unwrap().list().to_string();
        Ok(unsafe { write_out(buf, cap, &list) })
    })
}

fn identify_json(gallery_h: i64, engine_h: i64, img: &RgbImage, threshold_milli: i64, top_k: i64) -> Result<String, String> {
    let faces = run(engine_h, img)?;
    let threshold = gallery::threshold(threshold_milli);
    let g = gallery(gallery_h)?;
    let g = g.lock().unwrap();
    let list: Vec<Value> = faces
        .iter()
        .map(|f| {
            let mut v = face_json(f, img.width(), img.height(), false);
            let ranked = g.rank(&f.embedding, top_k.max(1) as usize);
            let best = ranked.first().filter(|m| m.similarity >= threshold);
            v["name"] = json!(best.map(|m| m.name.as_str()));
            v["similarity"] = json!(ranked.first().map(|m| m.similarity));
            v["candidates"] = json!(ranked.iter().map(Match::to_json).collect::<Vec<_>>());
            v
        })
        .collect();
    Ok(json!({ "width": img.width(), "height": img.height(), "threshold": gallery::threshold_json(threshold), "faces": list }).to_string())
}

/// Identifica cada rosto de uma imagem contra a galeria. Escreve JSON
/// `{width, height, threshold, faces: [{bbox, landmarks, score, name, similarity, candidates: [{name, similarity}]}]}`.
/// `name` é `null` quando a melhor similaridade fica abaixo do limiar (rosto desconhecido).
/// `threshold_milli` em milésimos (<= 0 usa o padrão); `top_k` candidatos por rosto.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_identify_file(
    gallery_h: i64,
    engine_h: i64,
    path: *const c_char,
    threshold_milli: i64,
    top_k: i64,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    guard(-1, || {
        let img = load_image(unsafe { text(path) }?)?;
        Ok(unsafe { write_out(buf, cap, &identify_json(gallery_h, engine_h, &img, threshold_milli, top_k)?) })
    })
}

/// Igual a `fface_identify_file`, para uma imagem codificada na memória.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_identify_bytes(
    gallery_h: i64,
    engine_h: i64,
    data: *const u8,
    len: i64,
    threshold_milli: i64,
    top_k: i64,
    buf: *mut u8,
    cap: i64,
) -> i64 {
    guard(-1, || {
        let img = decode_image(data, len)?;
        Ok(unsafe { write_out(buf, cap, &identify_json(gallery_h, engine_h, &img, threshold_milli, top_k)?) })
    })
}

// ---------------------------------------------------------------------------
// Visualização
// ---------------------------------------------------------------------------

/// Desenha os rostos de `faces_json` (a lista `faces` de detect/identify) sobre a
/// imagem `in_path` e salva em `out_path` (formato pela extensão). Caixas verdes
/// para rostos identificados ou só detectados, vermelhas para desconhecidos (`name: null`).
/// `font_path` vazio procura uma fonte do sistema; sem fonte, desenha sem os nomes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fface_draw_file(
    in_path: *const c_char,
    out_path: *const c_char,
    faces_json: *const c_char,
    font_path: *const c_char,
) -> i32 {
    status(|| {
        let mut img = load_image(unsafe { text(in_path) }?)?;
        let v: Value = serde_json::from_str(unsafe { text(faces_json) }?).map_err(|e| format!("invalid faces JSON: {e}"))?;
        let faces = match &v {
            Value::Array(a) => a.as_slice(),
            Value::Object(_) => v.get("faces").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default(),
            _ => return Err("faces must be a list".into()),
        };
        draw::draw(&mut img, faces, unsafe { text(font_path) }?);
        let out = unsafe { text(out_path) }?;
        if let Some(dir) = std::path::Path::new(out).parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        img.save(out).map_err(|e| format!("{out}: {e}"))
    })
}

/// Registra (ou remove, com NULL) o callback (Fogu 0.7+, `ffi.callback`) que
/// recebe as saídas das funções chamadas com `buf = NULL, cap = 0` (ver `cb.rs`).
#[unsafe(no_mangle)]
pub extern "C" fn fface_set_sink(sink: Option<cb::Sink>) -> i32 {
    cb::set_sink(sink);
    0
}
