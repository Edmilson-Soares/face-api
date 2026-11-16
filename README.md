# face — reconhecimento e identificação de faces para Fogu

Uma lib Rust (`foguface`) sobre o [insight-face-rs](https://crates.io/crates/insight-face-rs)
(ONNX Runtime, modelos InsightFace), mais um módulo Fogu (`face.fg`) que a usa via `fogu:ffi`.

- **Detecção**: caixa, 5 landmarks (olhos, nariz, cantos da boca) e confiança de cada rosto (SCRFD).
- **Reconhecimento**: embedding de 512 dimensões por rosto (ArcFace) e similaridade de cosseno.
- **Verificação 1:1**: duas fotos são da mesma pessoa?
- **Identificação 1:N**: galeria de pessoas cadastradas, salva/carregada em JSON;
  cada rosto de uma foto recebe um nome ou `nil` (desconhecido).
- **Visualização**: desenha caixas, pontos e nomes numa cópia da imagem.

```
foguface/            lib Rust (cdylib) com a API C
native/<plataforma>/ biblioteca nativa de cada sistema (gerada pelo build)
models/              modelos ONNX (baixados por models.ps1 / models.sh)
face.fg              wrapper Fogu: engine(), gallery(), draw(), similarity()
main.fg              exemplo curto
examples/            detectar.fg, comparar.fg, identificar.fg
images/              fotos de exemplo em domínio público (veja images/CREDITOS.md)
face.test.fg         testes
```

<!-- fogu:plataformas -->
## Plataformas, binários e pré-requisitos

Ao importar `face.fg`, a lib reconhece o sistema e a arquitetura
(`system.info.os`/`arch`) e já carrega o binário certo — `.dll` no Windows,
`.so` no Linux, `.dylib` no macOS. Se ele faltar, o programa não para no `import`:
o erro (com a lista de caminhos tentados) volta na primeira chamada que precisar da lib.
`nativeInfo()` mostra o que foi detectado: `{ platform, os, file, path, loaded, error }`.

| Plataforma | Arquivo | Neste repositório |
|---|---|---|
| `windows-x64` | `native/windows-x64/foguface.dll` | incluído |
| `windows-arm64` | `native/windows-arm64/foguface.dll` | — |
| `linux-x64` | `native/linux-x64/libfoguface.so` | incluído |
| `linux-arm64` | `native/linux-arm64/libfoguface.so` | incluído |
| `macos-x64` | `native/macos-x64/libfoguface.dylib` | — |
| `macos-arm64` | `native/macos-arm64/libfoguface.dylib` | — |

Ordem de busca: o caminho passado a `load(path)`, a variável `FOGU_FACE_LIB`,
`native/<plataforma>/`, `native/` e o build do cargo no projeto em execução, uma
cópia em `fogu_modules/` e os pacotes instalados com `fogu add` (`$FOGU_HOME/fogu_modules`).

**Pré-requisitos**

- **Windows / Linux**: nada além do binário (o ONNX Runtime vem embutido).
- **macOS**: o `.dylib` precisa ser compilado num Mac (`sh build.sh` lá): o ONNX Runtime de macOS linka os frameworks `Foundation` e `CoreML`, que só existem no SDK da Apple.
- Modelos ONNX em `models/` (`fogu run models`).

**Gerar os binários**

```sh
fogu run build:native        # Windows: native/windows-<arch>/ (build.ps1)
sh build.sh                  # Linux/macOS/WSL: o sistema atual
sh build.sh all              # Linux x64/arm64 e macOS x64/arm64 (cargo-zigbuild + zig)
```

## Callbacks do fogu:ffi (Fogu 0.7+, testado com o 0.7.4)

A camada nativa usa `ffi.callback` (veja `_native.fg`, igual em todas as libs deste repositório):

- **Sink de saída**: a lib recebe um callback uma vez (`*_set_sink`) e, chamada com
  `buf = nil, cap = 0`, entrega a resposta inteira por ele. As funções antigas, com buffer
  do chamador, continuam exportadas e funcionando (API C compatível).
- **Regras do fogu:ffi respeitadas**: um callback só roda na thread do interpretador, durante
  uma chamada FFI — as threads da lib nunca chamam o Fogu; elas enfileiram e a chamada
  `*_dispatch`/leitura entrega na thread de quem chamou. Um erro dentro de um callback derruba o
  programa, então o callback só guarda os dados e os handlers do usuário rodam depois, fora
  do frame nativo (com `?`, erros e `sleep` normais).
- **Esperas cooperativas**: uma chamada FFI bloqueante congela todas as tarefas `fg`, então as
  esperas longas são feitas em fatias de 10 ms com `time.delay(1ms)` entre elas (o mesmo
  padrão do pacote oficial `hardware`).

As respostas (detecção, embeddings, identificação) chegam inteiras pelo **sink** (`fface_set_sink`): antes, um buffer pequeno fazia a inferência rodar **duas vezes**; agora roda uma vez só, sem buffer de 1 MB "por garantia". As imagens em memória vão para a lib com `buffer.write` direto (antes eram copiadas byte a byte para uma lista).
- **Fogu 0.7.4**: o código compilado chama direto as funções FFI de assinatura numérica (sem
  alocação por chamada), então sondar status/eventos num laço é barato; `fogu .` roda o projeto
  (o script `start` do `fogu.json`); os tipos `int32`, `uint8`, `float64`... valem nas assinaturas.
<!-- /fogu:plataformas -->

## Build e execução

```sh
fogu run models              # baixa os modelos (~280 MB) — ou ./models.sh
fogu run build:native        # compila a lib — ou ./build.ps1 / ./build.sh
fogu run main.fg
fogu run detectar            # examples/detectar.fg    -> out/detectar.jpg
fogu run comparar            # examples/comparar.fg
fogu run identificar         # examples/identificar.fg -> out/identificar_*.jpg
fogu test
```

Requer Rust ≥ 1.85 (edition 2024). O ONNX Runtime é baixado pelo crate `ort` durante
o build e fica **embutido na DLL** (~25 MB): não há outra dependência para distribuir.
A biblioteca é procurada em `native/<plataforma>/` (`windows-x64`, `linux-arm64`, ...),
como no vpnclient; os caminhos (modelos, imagens, `native/`) são relativos à pasta
em que o programa roda.

## Exemplo

```fogu
import face from "./face.fg"

fn main() {
    const eng = face.engine()?                       // models/det_10g.onnx + models/w600k_r50.onnx
    const gal = face.gallery(eng)?
    gal.enroll("Obama", "images/gallery/obama.jpg")?
    gal.enroll("Biden", "images/gallery/biden.jpg")?

    const r = gal.identify("images/probe/situation_room.jpg")?
    for f in r.faces {
        if f.name != nil { print(f.name, f.similarity, f.bbox) }
    }
    face.draw("images/probe/situation_room.jpg", "out/grupo.jpg", r.faces)?
    return true, nil
}
main()
```

Resultado com as fotos de `images/` (limiar 0.40):

| comparação | similaridade | |
|---|---|---|
| Biden 2021 × Biden 2013 | 0.67 | mesma pessoa |
| Harris 2021 × Harris 2017 | 0.80 | mesma pessoa |
| Obama × Biden | −0.03 | diferentes |
| foto de grupo (12 rostos) | Obama 0.70, Biden 0.45 | os outros 10: desconhecidos (≤ 0.14) |

## API Fogu

`face.engine(options)` → `FaceEngine`. Opções: `det_model` (`"models/det_10g.onnx"`),
`rec_model` (`"models/w600k_r50.onnx"`), `idle_timeout` (segundos sem uso até liberar
a memória dos modelos, que voltam sozinhos na próxima chamada; `0` = nunca).

| `FaceEngine` | |
|---|---|
| `detect(path, withEmbedding = false)` | `{width, height, faces: [{score, bbox: {x, y, width, height}, landmarks: [[x, y] ×5], embedding?}]}` em pixels |
| `detectBytes(data, withEmbedding)` | igual, com a imagem (JPEG, PNG, WebP...) em memória (`bytes`, lista ou str) |
| `embedding(path)` | 512 números do maior rosto da foto |
| `compare(a, b, threshold = 0.40)` | `{similarity, same, threshold}` entre os maiores rostos das duas fotos |
| `close()` | libera os modelos |

`face.gallery(engine, path = nil)` → `Gallery` vazia ou carregada de um JSON.

| `Gallery` | |
|---|---|
| `enroll(name, path)` / `enrollBytes(name, data)` | cadastra o maior rosto da foto; várias fotos do mesmo nome somam amostras. Devolve o nº de amostras |
| `enrollEmbedding(name, embedding)` | cadastra um embedding já calculado |
| `identify(path, threshold = 0.40, topK = 3)` / `identifyBytes(...)` | cada rosto com `name` (`nil` = desconhecido), `similarity` e `candidates: [{name, similarity}]` |
| `list()` | `[{name, samples}]` |
| `remove(name)` | `true` se existia |
| `save(path)` | JSON com nomes e embeddings (as fotos não são guardadas) |
| `close()` | |

A similaridade com uma pessoa é a maior entre as amostras dela. Também há
`face.similarity(a, b)` (dois embeddings), `face.draw(input, output, faces, font = "")`
(verde = identificado/detectado, vermelho = desconhecido; rótulo `label` ou `name` +
similaridade; sem `font` procura Arial/DejaVu do sistema), `face.version()`,
`face.platform()` e `face.load(path)`.

**Limiar**: 0.40 é um bom ponto de partida para o `w600k_r50`. Aumente (0.5–0.6) para
menos falsos positivos (controle de acesso), diminua (0.3) para achar a mesma pessoa
em fotos ruins/de perfil. Rostos pequenos (< ~60 px) e de perfil dão similaridades menores.

## API C (para usar de outras linguagens)

Handles são `i64`. Funções de status retornam `0`/`-1` (mensagem em `fface_last_error`).
Saída vai para um buffer do chamador; o retorno é o tamanho escrito ou `-(tamanho necessário)`.
Números reais trafegam em JSON; limiares em milésimos (`400` = 0.40; `<= 0` = padrão).

```c
int64_t fface_version(char *buf, int64_t cap);
int64_t fface_last_error(char *buf, int64_t cap);
int64_t fface_engine_new(const char *options_json);                 // handle ou -1
int32_t fface_engine_free(int64_t engine);
int64_t fface_detect_file(int64_t engine, const char *path, int32_t with_embedding, char *buf, int64_t cap);
int64_t fface_detect_bytes(int64_t engine, const uint8_t *data, int64_t len, int32_t with_embedding, char *buf, int64_t cap);
int64_t fface_compare_files(int64_t engine, const char *a, const char *b, int64_t threshold_milli, char *buf, int64_t cap);
int64_t fface_similarity(const char *a_json, const char *b_json, char *buf, int64_t cap);
int64_t fface_gallery_new(void);
int64_t fface_gallery_load(const char *path);
int32_t fface_gallery_save(int64_t gallery, const char *path);
int32_t fface_gallery_free(int64_t gallery);
int64_t fface_gallery_enroll_file(int64_t gallery, int64_t engine, const char *name, const char *path);   // nº de amostras
int64_t fface_gallery_enroll_bytes(int64_t gallery, int64_t engine, const char *name, const uint8_t *data, int64_t len);
int64_t fface_gallery_enroll_embedding(int64_t gallery, const char *name, const char *embedding_json);
int64_t fface_gallery_remove(int64_t gallery, const char *name);    // 1 removida, 0 não existia
int64_t fface_gallery_list(int64_t gallery, char *buf, int64_t cap);
int64_t fface_identify_file(int64_t gallery, int64_t engine, const char *path, int64_t threshold_milli, int64_t top_k, char *buf, int64_t cap);
int64_t fface_identify_bytes(int64_t gallery, int64_t engine, const uint8_t *data, int64_t len, int64_t threshold_milli, int64_t top_k, char *buf, int64_t cap);
int32_t fface_draw_file(const char *in_path, const char *out_path, const char *faces_json, const char *font_path);
```

## Notas

- **Licença dos modelos**: o código (insight-face-rs, esta lib) é MIT, mas os modelos
  pré-treinados do InsightFace (`buffalo_l`) são liberados **só para pesquisa / uso não
  comercial**. Para uso comercial, licencie com o InsightFace ou treine/obtenha outro
  modelo SCRFD + ArcFace compatível (basta apontar `det_model` / `rec_model`).
- **Dados biométricos**: embeddings de rosto são dados pessoais sensíveis (LGPD, art. 11).
  Guarde a galeria com o mesmo cuidado que guardaria as fotos e tenha base legal/consentimento
  para cadastrar pessoas.
- O `insight-face-rs` 4.2.1 não compila com o `ort` 2.0.0-rc.15; o `Cargo.toml` fixa o rc.13.
- Só CPU por padrão. Para CUDA, ligue a feature `cuda` do insight-face-rs (precisa de CUDA/cuDNN).
- Se a resposta não couber no buffer, o wrapper roda a inferência de novo com um buffer
  maior; por isso as chamadas de detecção/identificação já começam com 1 MB.
