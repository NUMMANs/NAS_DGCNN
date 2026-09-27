# run_prep.ps1 -- S3DIS density-domain prep / calibration runner (ASCII only)
#
# Why plain ASCII: Windows PowerShell 5.1 decodes .ps1 as the system ANSI
# codepage (GBK on zh-CN). Non-ASCII comments get mis-decoded and can break
# parsing. Keep this file 7-bit clean.
#
# Why clear the proxy vars: the sandboxed proxy answers 502 for these hosts.
# Direct connections are required (the dataset was mirrored over direct HTTPS).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File nas_ptseg\run_prep.ps1 -Mode list
#   powershell -ExecutionPolicy Bypass -File nas_ptseg\run_prep.ps1 -Mode build
#   powershell -ExecutionPolicy Bypass -File nas_ptseg\run_prep.ps1 -Mode calibrate
#   powershell -ExecutionPolicy Bypass -File nas_ptseg\run_prep.ps1 -Mode pareto

param(
    [ValidateSet("list", "build", "calibrate", "pareto", "dense-diag", "edge-null", "k-sweep", "seed-noise", "geom-stats")]
    [string]$Mode = "calibrate",
    [string]$Area = "Area_1",
    [string]$Room = "conferenceRoom_1",
    [string]$Report = "",
    [string]$Extra = ""
)

$ErrorActionPreference = "Stop"

# --- isolate from sandbox proxy -------------------------------------------------
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

# keep BLAS/OpenMP from oversubscribing during repeated proxy evaluations
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

# --- resolve paths --------------------------------------------------------------
$Py = "D:\anaconda\envs\openevolve\python.exe"
if (-not (Test-Path $Py)) {
    throw "python not found: $Py"
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Prep = Join-Path $PSScriptRoot "prep.py"
if (-not (Test-Path $Prep)) {
    throw "prep.py not found: $Prep"
}
if ([string]::IsNullOrWhiteSpace($Report)) {
    $Report = Join-Path $ProjectRoot ("_" + $Mode + ".txt")
}

# --- build argument list --------------------------------------------------------
$PrepArgs = @($Prep, "--$Mode", "--area", $Area, "--room", $Room, "--report", $Report)
if (-not [string]::IsNullOrWhiteSpace($Extra)) {
    $PrepArgs += ($Extra -split "\s+")
}

Write-Host "[run_prep] mode    = $Mode"
Write-Host "[run_prep] python  = $Py"
Write-Host "[run_prep] report  = $Report"
Write-Host "[run_prep] room    = $Area/$Room"
if (-not [string]::IsNullOrWhiteSpace($Extra)) {
    Write-Host "[run_prep] extra   = $Extra"
}
Write-Host ""

# Python writes the report itself in UTF-8; do not rely on PS stdout capture.
& $Py @PrepArgs
$code = $LASTEXITCODE

Write-Host ""
Write-Host "[run_prep] exit code = $code"
if ($code -eq 0) {
    Write-Host "[run_prep] report written to: $Report"
} else {
    Write-Host "[run_prep] FAILED. See report for the tail of the log."
}
exit $code
