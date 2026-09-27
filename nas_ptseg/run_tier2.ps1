# run_tier2.ps1 -- Tier-2 real DGCNN short-training runner (ASCII only)
#
# Why plain ASCII: Windows PowerShell 5.1 decodes .ps1 as the system ANSI codepage
# (GBK on zh-CN); non-ASCII comments get mis-decoded and can break parsing.
#
# Why clear the proxy vars: the sandboxed proxy answers 502 for these hosts.
#
# Why CUDA_VISIBLE_DEVICES=1: standing project convention (see .workbuddy/memory).
# It must be set BEFORE python imports torch, so it is set here, not inside python.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File nas_ptseg\run_tier2.ps1 -Mode single -TrainDomain rho0 -K 16 -Epochs 200
#   powershell -ExecutionPolicy Bypass -File nas_ptseg\run_tier2.ps1 -Mode window -TrainDomain rho0 -Epochs 200
#   powershell -ExecutionPolicy Bypass -File nas_ptseg\run_tier2.ps1 -Mode matrix -Epochs 200
#   powershell -ExecutionPolicy Bypass -File nas_ptseg\run_tier2.ps1 -Mode analyze

param(
    [ValidateSet("single", "window", "matrix", "analyze", "grid", "crop-control", "alpha-sweep")]
    [string]$Mode = "single",
    [string]$TrainDomain = "rho0",
    [int]$K = 16,
    [int]$Epochs = 200,
    [double]$HoldoutFrac = 0.3,
    [int]$VerboseEvery = 10,
    [string]$OutDir = "",
    [string]$Report = "",
    [string]$Extra = ""
)

$ErrorActionPreference = "Stop"

# --- isolate from sandbox proxy -------------------------------------------------
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

# --- pin the training GPU -------------------------------------------------------
# HARD CONSTRAINT (user, 2026-09-23): every GPU operation must run on physical GPU 1,
# and must never squeeze out another program's resources. Physical GPU 0 is usually
# busy with the user's own jobs, so defaulting to device 0 would be destructive.
# This env var is only the outer layer of the guard: nas_ptseg/gpu.py re-pins it at
# package import time (before torch reads it) and refuses to start when free memory
# is insufficient. Do not remove either layer.
$env:CUDA_VISIBLE_DEVICES = "1"

# --- do NOT set OMP/MKL/OPENBLAS thread counts here -----------------------------
# Setting OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS makes this env
# die with a NATIVE crash (exit 0xC0000409, STATUS_STACK_BUFFER_OVERRUN) within the
# first ~20 epochs of training, with no Python traceback. Confirmed by A/B test:
# the identical command run directly (without these vars) completes fine.
# Leave the thread counts at their defaults.

# --- resolve paths --------------------------------------------------------------
$Py = "D:\anaconda\envs\lidar310\python.exe"
if (-not (Test-Path $Py)) {
    throw "python not found: $Py"
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $PSScriptRoot "tier2_dgcnn.py"
if (-not (Test-Path $Script)) {
    throw "tier2_dgcnn.py not found: $Script"
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $ProjectRoot "tier2_results"
}
if ([string]::IsNullOrWhiteSpace($Report)) {
    $Report = Join-Path $ProjectRoot ("_tier2_" + $Mode + ".txt")
}

# --- build argument list --------------------------------------------------------
$PyArgs = @($Script)
switch ($Mode) {
    "single"  { $PyArgs += @("--train-domain", $TrainDomain, "--k", $K,
                             "--epochs", $Epochs, "--holdout-frac", $HoldoutFrac,
                             "--verbose-every", $VerboseEvery, "--out-dir", $OutDir) }
    "window"  { $PyArgs += @("--window-probe", "--train-domain", $TrainDomain,
                             "--epochs", $Epochs, "--holdout-frac", $HoldoutFrac,
                             "--verbose-every", $VerboseEvery, "--out-dir", $OutDir) }
    "matrix"  { $PyArgs += @("--matrix", "--epochs", $Epochs,
                             "--holdout-frac", $HoldoutFrac,
                             "--verbose-every", $VerboseEvery, "--out-dir", $OutDir) }
    "analyze" { $PyArgs += @("--analyze", "--out-dir", $OutDir) }
    "grid"    { $PyArgs += @("--grid", "--epochs", $Epochs,
                             "--holdout-frac", $HoldoutFrac,
                             "--verbose-every", $VerboseEvery, "--out-dir", $OutDir) }
    "crop-control" {
                $PyArgs += @("--crop-control", "--train-domain", $TrainDomain,
                             "--epochs", $Epochs, "--holdout-frac", $HoldoutFrac,
                             "--verbose-every", $VerboseEvery, "--out-dir", $OutDir) }
    "alpha-sweep"  {
                $PyArgs += @("--alpha-sweep", "--epochs", $Epochs,
                             "--holdout-frac", $HoldoutFrac,
                             "--verbose-every", $VerboseEvery, "--out-dir", $OutDir) }
}

if (-not [string]::IsNullOrWhiteSpace($Extra)) {
    $PyArgs += ($Extra -split "\s+")
}

Write-Host "[tier2] mode    = $Mode"
Write-Host "[tier2] python  = $Py"
Write-Host "[tier2] GPU     = CUDA_VISIBLE_DEVICES=1"
Write-Host "[tier2] outdir  = $OutDir"
Write-Host "[tier2] report  = $Report"
Write-Host ""

# Ask python to write the UTF-8 report itself. Do NOT pipe stdout through
# PowerShell: it re-encodes child output with the ANSI codepage and turns
# Chinese into mojibake (a known trap on this machine).
$PyArgs += @("--log-file", $Report)

Push-Location $ProjectRoot
try {
    & $Py @PyArgs
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "[tier2] exit code = $code"
exit $code
