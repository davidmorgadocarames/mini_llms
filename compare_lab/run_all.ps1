# Fase D, Etapa 1 -- run everything for Cracked-D unattended, end to end.
#
# Four stages, in order, in a single process:
#   1. pretraining data  (tokenizer + 1.2B tokens streamed to binaries)  ~40 min
#   2. fine-tuning data  (smol-smoltalk, ~150k conversations)            ~5 min
#   3. pretraining       (prefix-LM, 1 epoch, 4577 steps)                ~6.3 h
#   4. fine-tuning       (chat SFT, 20000 steps)                         ~40 min
#
# No admin rights needed. Relaunching this same script resumes: finished stages
# are skipped, and an interrupted pretrain/fine-tune continues from its last
# checkpoint (written every 5 min). Only stage 1 restarts from scratch if
# interrupted.
#
#   Pause (loses nothing):  create a STOP file in the active stage's run dir,
#                           e.g. compare_lab\runs\pretrain_cracked\STOP
#   Progress:               python -m compare_lab.status
#   Log:                    compare_lab\runs\pipeline_cracked\pipeline.log
#
# See compare_lab\PAUSAR_Y_REANUDAR.md for the full pause/resume story.

# NOT "Stop": piping a native command's stderr through 2>&1 wraps every line in
# an ErrorRecord in PowerShell 5.1, so a harmless Python warning on stderr (the
# urllib3 version notice) would abort the script before training even starts.
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:HF_HUB_DISABLE_XET = "1"   # the Xet backend hangs on this network (CLAUDE.md, Fase A)
$env:PYTHONPATH = $root
$env:PYTHONUNBUFFERED = "1"     # live log instead of buffered chunks

$logDir = Join-Path $root "compare_lab\runs\pipeline_cracked"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "pipeline.log"

# Keep the machine awake for as long as this script runs, WITHOUT touching the
# user's power plan: ES_CONTINUOUS | ES_SYSTEM_REQUIRED tells Windows the system
# is in use. The display may still sleep (deliberate -- it is an overnight run),
# and the flag is dropped automatically when this process exits. No admin needed.
try {
    $sig = @'
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
    $power = Add-Type -MemberDefinition $sig -Name 'Power' -Namespace 'Win32' -PassThru
    $null = $power::SetThreadExecutionState([uint32]'0x80000001')
    $awake = "si (se revierte al cerrar esta ventana)"
} catch {
    $awake = "NO se pudo activar -- comprueba que la suspension este desactivada"
}

Write-Host ""
Write-Host "  FASE D - ETAPA 1 - Cracked-D, ejecucion completa" -ForegroundColor Cyan
Write-Host "  ------------------------------------------------"
Write-Host "  raiz:            $root"
Write-Host "  log:             $log"
Write-Host "  progreso:        python -m compare_lab.status"
Write-Host "  evitar suspension: $awake"
Write-Host ""
Write-Host "  Duracion estimada total: ~8 horas." -ForegroundColor Yellow
Write-Host "  Puedes cerrar sesion o irte; NO cierres esta ventana." -ForegroundColor Yellow
Write-Host "  Para pausar sin perder nada, crea un fichero STOP (ver PAUSAR_Y_REANUDAR.md)."
Write-Host ""
Write-Host "  NOTA: veras lineas rojas de 'NativeCommandError' con un aviso de urllib3." -ForegroundColor DarkGray
Write-Host "  NO es un fallo: PowerShell pinta de rojo todo lo que Python escribe en" -ForegroundColor DarkGray
Write-Host "  stderr, incluidos los avisos inofensivos. Lo que importa son las lineas" -ForegroundColor DarkGray
Write-Host "  'stage N/4' y el progreso. Si algo falla de verdad, lo dira al final." -ForegroundColor DarkGray
Write-Host ""

$start = Get-Date
# Deliberately NOT piped through Tee-Object. tqdm redraws its progress bar on
# stderr with carriage returns; PowerShell turns each redraw into a separate
# ErrorRecord line, so piping made the bar invisible for the whole 8-hour run.
# Every stage writes its own log from Python instead:
#   compare_lab\runs\pipeline_cracked\pipeline.log   (stage transitions)
#   compare_lab\runs\pretrain_cracked\pretrain_cracked.log
#   compare_lab\runs\finetune_cracked\finetune_cracked.log
python -m compare_lab.train.run_pipeline --arch cracked
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $start

Write-Host ""
Write-Host "  ------------------------------------------------"
Write-Host ("  Tiempo total: {0:hh\:mm\:ss}" -f $elapsed)
if ($code -eq 0) {
    Write-Host "  TERMINADO. Revisa el estado con: python -m compare_lab.status" -ForegroundColor Green
    Write-Host "  Checkpoints en: compare_lab\checkpoints\cracked\" -ForegroundColor Green
} else {
    Write-Host "  TERMINO CON CODIGO $code -- mira el final del log." -ForegroundColor Red
    Write-Host "  Relanzar este mismo script reanuda donde se quedo." -ForegroundColor Red
}
Write-Host ""
Write-Host "  Esta ventana queda abierta a proposito. Pulsa Enter para cerrarla."
$null = Read-Host
