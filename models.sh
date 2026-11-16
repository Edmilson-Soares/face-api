#!/bin/sh
# Baixa os modelos ONNX do InsightFace (pacote buffalo_l) para models/:
#   det_10g.onnx    detecção (SCRFD)
#   w600k_r50.onnx  reconhecimento (ArcFace, embeddings de 512 dimensões)
# Licença dos modelos: só para pesquisa/uso não comercial (veja README).
set -e
cd "$(dirname "$0")"
if [ -f models/det_10g.onnx ] && [ -f models/w600k_r50.onnx ]; then
    echo "modelos já estão em models/"
    exit 0
fi
tmp="${TMPDIR:-/tmp}/foguface-models"
mkdir -p "$tmp" models
if [ ! -f "$tmp/buffalo_l.zip" ]; then
    echo "baixando buffalo_l.zip (~280 MB)..."
    curl -fL -o "$tmp/buffalo_l.zip" https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
fi
echo "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f  $tmp/buffalo_l.zip" | sha256sum -c - >/dev/null \
    || { rm -f "$tmp/buffalo_l.zip"; echo "hash inesperado do buffalo_l.zip" >&2; exit 1; }
unzip -o -j "$tmp/buffalo_l.zip" det_10g.onnx w600k_r50.onnx -d models
echo "models/det_10g.onnx e models/w600k_r50.onnx prontos"
