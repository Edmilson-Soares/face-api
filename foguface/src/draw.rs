//! Desenha os rostos (caixa, landmarks e rótulo) numa cópia da imagem.

use ab_glyph::{FontVec, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;
use serde_json::Value;

const KNOWN: Rgb<u8> = Rgb([40, 200, 80]);
const UNKNOWN: Rgb<u8> = Rgb([230, 50, 50]);
const POINT: Rgb<u8> = Rgb([255, 210, 0]);
const WHITE: Rgb<u8> = Rgb([255, 255, 255]);

/// Fontes tentadas quando o chamador não passa uma.
const SYSTEM_FONTS: &[&str] = &[
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
];

fn load_font(path: &str) -> Option<FontVec> {
    let candidates: Vec<&str> = if path.is_empty() { SYSTEM_FONTS.to_vec() } else { vec![path] };
    candidates.into_iter().find_map(|p| FontVec::try_from_vec(std::fs::read(p).ok()?).ok())
}

fn num(v: &Value, key: &str) -> f32 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0) as f32
}

/// `faces` é a lista `faces` devolvida por detect/identify (coordenadas em pixels).
/// O rótulo vem de `label`, senão de `name` (+ similaridade); `name: null` vira "?" em vermelho.
pub fn draw(img: &mut RgbImage, faces: &[Value], font_path: &str) {
    let font = load_font(font_path);
    let thick = (img.width().max(img.height()) / 400).max(2) as i32;
    let scale = PxScale::from((img.height() as f32 / 40.0).clamp(14.0, 48.0));

    for f in faces {
        let Some(b) = f.get("bbox") else { continue };
        let (x, y, w, h) = (num(b, "x"), num(b, "y"), num(b, "width"), num(b, "height"));
        let identified = f.get("name").is_some();
        let name = f.get("name").and_then(Value::as_str);
        let color = if identified && name.is_none() { UNKNOWN } else { KNOWN };

        for t in 0..thick {
            let r = Rect::at(x as i32 - t, y as i32 - t).of_size((w as u32 + 2 * t as u32).max(1), (h as u32 + 2 * t as u32).max(1));
            draw_hollow_rect_mut(img, r, color);
        }
        if let Some(points) = f.get("landmarks").and_then(Value::as_array) {
            for p in points {
                let px = p.get(0).and_then(Value::as_f64).unwrap_or(0.0) as i32;
                let py = p.get(1).and_then(Value::as_f64).unwrap_or(0.0) as i32;
                draw_filled_circle_mut(img, (px, py), thick, POINT);
            }
        }

        let label = match (f.get("label").and_then(Value::as_str), name) {
            (Some(l), _) => l.to_string(),
            (None, Some(n)) => match f.get("similarity").and_then(Value::as_f64) {
                Some(s) => format!("{n} {s:.2}"),
                None => n.to_string(),
            },
            (None, None) if identified => "?".to_string(),
            (None, None) => continue,
        };
        let Some(font) = &font else { continue };
        let (tw, th) = text_size(scale, font, &label);
        let pad = 4;
        let ty = (y as i32 - th as i32 - 2 * pad - thick).max(0);
        draw_filled_rect_mut(img, Rect::at(x as i32 - thick, ty).of_size(tw + 2 * pad as u32, th + 2 * pad as u32), color);
        draw_text_mut(img, WHITE, x as i32 - thick + pad, ty + pad, scale, font, &label);
    }
}
