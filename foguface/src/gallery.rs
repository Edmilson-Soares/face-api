//! Galeria de pessoas conhecidas para identificação 1:N.
//!
//! Cada pessoa guarda as amostras (embeddings) cadastradas; a similaridade de um
//! rosto com a pessoa é a maior similaridade de cosseno entre ele e as amostras.

use insight_face_rs::FaceEmbedding;
use serde_json::{Value, json};

use crate::embedding_from_json;

/// Limiar padrão de similaridade de cosseno para considerar a mesma pessoa
/// (bom ponto de partida para os modelos ArcFace w600k do InsightFace).
pub const DEFAULT_THRESHOLD: f32 = 0.40;

/// Limiar em milésimos vindo da API C; `<= 0` usa o padrão.
pub fn threshold(milli: i64) -> f32 {
    if milli <= 0 { DEFAULT_THRESHOLD } else { milli as f32 / 1000.0 }
}

/// Limiar para o JSON, sem o ruído do f32 (0.4 em vez de 0.4000000059604645).
pub fn threshold_json(t: f32) -> f64 {
    (t as f64 * 1000.0).round() / 1000.0
}

struct Person {
    name: String,
    samples: Vec<FaceEmbedding>,
}

#[derive(Default)]
pub struct Gallery {
    people: Vec<Person>,
}

pub struct Match {
    pub name: String,
    pub similarity: f32,
}

impl Match {
    pub fn to_json(&self) -> Value {
        json!({ "name": self.name, "similarity": self.similarity })
    }
}

impl Gallery {
    /// Adiciona uma amostra à pessoa (criando-a se preciso). Retorna quantas amostras ela tem.
    pub fn enroll(&mut self, name: &str, embedding: FaceEmbedding) -> usize {
        let embedding = embedding.normalize();
        match self.people.iter_mut().find(|p| p.name == name) {
            Some(p) => {
                p.samples.push(embedding);
                p.samples.len()
            }
            None => {
                self.people.push(Person { name: name.to_string(), samples: vec![embedding] });
                1
            }
        }
    }

    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.people.len();
        self.people.retain(|p| p.name != name);
        self.people.len() != before
    }

    pub fn list(&self) -> Value {
        json!(self.people.iter().map(|p| json!({ "name": p.name, "samples": p.samples.len() })).collect::<Vec<_>>())
    }

    /// As `top_k` pessoas mais parecidas com `embedding`, da maior para a menor similaridade.
    pub fn rank(&self, embedding: &FaceEmbedding, top_k: usize) -> Vec<Match> {
        let mut out: Vec<Match> = self
            .people
            .iter()
            .map(|p| Match {
                name: p.name.clone(),
                similarity: p.samples.iter().map(|s| s.cosine_similarity(embedding)).fold(f32::MIN, f32::max),
            })
            .collect();
        out.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        out.truncate(top_k);
        out
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let people: Vec<Value> = self
            .people
            .iter()
            .map(|p| json!({ "name": p.name, "embeddings": p.samples.iter().map(|s| s.0.to_vec()).collect::<Vec<_>>() }))
            .collect();
        let doc = json!({ "format": "foguface-gallery", "version": 1, "people": people });
        std::fs::write(path, doc.to_string()).map_err(|e| format!("{path}: {e}"))
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let doc: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: invalid JSON: {e}"))?;
        let mut g = Gallery::default();
        for p in doc.get("people").and_then(Value::as_array).ok_or(format!("{path}: not a foguface gallery"))? {
            let name = p.get("name").and_then(Value::as_str).ok_or(format!("{path}: person without name"))?;
            for e in p.get("embeddings").and_then(Value::as_array).into_iter().flatten() {
                g.enroll(name, embedding_from_json(e)?);
            }
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(i: usize) -> FaceEmbedding {
        let mut v = [0f32; 512];
        v[i] = 1.0;
        FaceEmbedding(v)
    }

    #[test]
    fn rank_uses_best_sample() {
        let mut g = Gallery::default();
        g.enroll("a", axis(0));
        g.enroll("a", axis(1));
        g.enroll("b", axis(2));
        let r = g.rank(&axis(1), 5);
        assert_eq!(r[0].name, "a");
        assert!((r[0].similarity - 1.0).abs() < 1e-6);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut g = Gallery::default();
        g.enroll("a", axis(3));
        let path = std::env::temp_dir().join("foguface_gallery_test.json");
        let path = path.to_str().unwrap();
        g.save(path).unwrap();
        let mut g2 = Gallery::load(path).unwrap();
        assert_eq!(g2.rank(&axis(3), 1)[0].name, "a");
        assert!(g2.remove("a"));
        assert!(g2.people.is_empty());
        std::fs::remove_file(path).ok();
    }
}
