# Round 6: sweep z on the REMAINING FOUR layers (2026-09-25/26).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# WHY: round 5 found a STRICT Pareto improvement by lowering one layer's z:
#   voxel15_z16,voxel20_z32,voxel25_z48,voxel35_z64,voxel50_z96@gate  0.4180  65.4ms
#   -> voxel15_z16,voxel20_z32,voxel25_z16,voxel35_z64,voxel50_z96@gate
#                                                                     0.4193  39.5ms
#   (more accurate AND 40% cheaper)
# Round 6 applies the same "lower the z" move to each of the other layers, one
# at a time, to see if it generalises or if v25 was special.
#
# Includes a REVERSE control (v25 z16 -> z32) to guard against the trap of
# concluding "lowering z always helps" from a single positive result.
#
# Uses arch_loop.measure() for multi-round retry (native crash has struck 6+ times).

param(
    [int]$Epochs = 300,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\openevolve\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_evolve6.log" }

$env:CUDA_VISIBLE_DEVICES = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

Push-Location $ProjectDir
try {
    & $PyExe -W ignore -m nas_ptseg.arch_loop `
        --epochs $Epochs --n-train 12 --n-val 12 `
        --out-dir (Join-Path $ProjectDir "data\multibranch_evolved6") `
        --log-file $Log `
        --archs `
        "voxel15_z16,voxel20_z16,voxel25_z16,voxel35_z64,voxel50_z96@gate" `
        "voxel15_z16,voxel20_z24,voxel25_z16,voxel35_z64,voxel50_z96@gate" `
        "voxel15_z16,voxel20_z32,voxel25_z16,voxel35_z32,voxel50_z96@gate" `
        "voxel15_z16,voxel20_z32,voxel25_z16,voxel35_z48,voxel50_z96@gate" `
        "voxel15_z16,voxel20_z32,voxel25_z32,voxel35_z64,voxel50_z96@gate"
    $code = $LASTEXITCODE
    Write-Host "[ev6] arch_loop exit=$code"
    Add-Content -Path $Log -Value "[ev6] arch_loop exit=$code" -Encoding UTF8
} finally {
    Pop-Location
}
