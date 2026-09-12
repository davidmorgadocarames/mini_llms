# Launch the Fase D pretraining data preparation in its own window.
#
# Kept as a script rather than a one-liner because the nested quoting needed by
# Start-Process -ArgumentList is easy to get subtly wrong (and did fail once,
# silently, leaving no process and no status.json).
#
# This stage streams ~1.2B tokens of SmolLM-Corpus, trains the shared tokenizer
# and writes the binaries plus the frozen prefix-LM plan. It is the ONE stage
# that cannot resume mid-way: interrupted, it starts over.
#
#   Progress:  python -m compare_lab.status
#   Log:       compare_lab/runs/data_prep_pretrain/data_prep.log

# NOT "Stop": piping a native command's stderr through 2>&1 wraps every line in
# an ErrorRecord in PowerShell 5.1, so a harmless Python warning on stderr (the
# urllib3 version notice) would abort the whole script before training starts.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:HF_HUB_DISABLE_XET = "1"   # Xet backend hangs on this network (see CLAUDE.md, Fase A)
$env:PYTHONPATH = $root
$env:PYTHONUNBUFFERED = "1"     # so the log updates live instead of in chunks

$logDir = Join-Path $root "compare_lab\runs\data_prep_pretrain"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "data_prep.log"

Write-Host "Fase D - preparacion de datos de preentrenamiento" -ForegroundColor Cyan
Write-Host "  raiz: $root"
Write-Host "  log:  $log"
Write-Host "  progreso: python -m compare_lab.status"
Write-Host ""

python -m compare_lab.data.prepare_pretrain 2>&1 | Tee-Object -FilePath $log

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "TERMINADO correctamente. Siguiente paso: el preentrenamiento." -ForegroundColor Green
} else {
    Write-Host "FALLO (codigo $LASTEXITCODE). Mira el final del log." -ForegroundColor Red
}
Write-Host "Esta ventana queda abierta para que veas el resultado."
