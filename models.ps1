# Baixa os modelos ONNX do InsightFace (pacote buffalo_l) para models/:
#   det_10g.onnx    detecção (SCRFD)
#   w600k_r50.onnx  reconhecimento (ArcFace, embeddings de 512 dimensões)
# Licença dos modelos: só para pesquisa/uso não comercial (veja README).
$ErrorActionPreference = "Stop"
$out = "$PSScriptRoot\models"
if ((Test-Path "$out\det_10g.onnx") -and (Test-Path "$out\w600k_r50.onnx")) {
    Write-Host "modelos já estão em models/"
    exit 0
}
$zip = Join-Path $env:TEMP "buffalo_l.zip"
$dir = Join-Path $env:TEMP "buffalo_l"
if (-not (Test-Path $zip)) {
    Write-Host "baixando buffalo_l.zip (~280 MB)..."
    Invoke-WebRequest "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -OutFile $zip
}
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
if ($hash -ne "80FFE37D8A5940D59A7384C201A2A38D4741F2F3C51EEF46EBB28218A7B0CA2F") {
    Remove-Item $zip
    throw "hash inesperado do buffalo_l.zip: $hash"
}
Expand-Archive $zip -DestinationPath $dir -Force
New-Item -ItemType Directory -Force $out | Out-Null
Copy-Item "$dir\det_10g.onnx", "$dir\w600k_r50.onnx" $out -Force
Write-Host "models/det_10g.onnx e models/w600k_r50.onnx prontos"
