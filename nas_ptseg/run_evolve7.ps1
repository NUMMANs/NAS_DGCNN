# Round 7: the FEATURE WIDTH axis on voxel branches (2026-09-26).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# WHY this is the right new axis: rounds 3-6 exhausted the scale/branch axis
# (marginal gain fell from +0.0033 to +0.0001, below sigma -- same convergence
# pattern as the old k axis). The feature width `feat` has NEVER been tested on
# voxel branches. For the points family it bought +0.019 (but at 2x params AND
# 2x wall time). Voxel branches are 14x cheaper per step, so the same capacity
# may be much cheaper to buy here -- and feat=32 tests the reverse: can we get
# cheaper without losing accuracy?
#
# Verified before spending GPU: feat really does reach the voxel branches
#   feat=32  -> 235,701 params
#   feat=64  -> 755,925 params   (champion)
#   feat=128 -> 2,810,133 params
# all with full gradient coverage.
#
# Uses arch_loop.measure() for multi-round retry (native crash has struck 6+ times).

param(
    [int]$Epochs = 300,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_evolve7.log" }

$env:CUDA_VISIBLE_DEVICES = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

Push-Location $ProjectDir
try {
    & $PyExe -W ignore -m nas_ptseg.multibranch_exp `
        --k 48 --epochs $Epochs --seeds 12345 23456 --n-train 12 --n-val 12 `
        --only `
        "voxel15_z16,voxel20_z32,voxel25_z16,voxel35_z64,voxel50_z96@gate|feat=128" `
        "voxel15_z16,voxel20_z32,voxel25_z16,voxel35_z64,voxel50_z96@gate|feat=32" `
        "voxel15_z16,voxel20_z32,voxel50_z96@gate|feat=128" `
        "voxel15_z16,voxel20_z32,voxel35_z64,voxel50_z96@gate|feat=128" `
        --out-dir (Join-Path $ProjectDir "data\multibranch_evolved7") `
        --log-file $Log
    $code = $LASTEXITCODE
    Write-Host "[ev7] exit=$code"
    Add-Content -Path $Log -Value "[ev7] exit=$code" -Encoding UTF8
} finally {
    Pop-Location
}
