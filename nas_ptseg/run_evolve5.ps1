# Round 5: leave-one-out ablation of the 5-branch champion (2026-09-25).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# WHY: round 4 gave the most accurate architecture yet
#     voxel15_z16,voxel20_z32,voxel25_z48,voxel35_z64,voxel50_z96@gate   val 0.4180
# but at 65.4 ms it is 2.1x the cost of the 4-branch (0.4147 @ 30.8 ms).
# The open question is WHICH BRANCH EARNS ITS COST. Leave-one-out answers it:
# drop each layer in turn; the drop that hurts least identifies the dispensable one.
# Then the 5-branch can be rebuilt cheaper without losing accuracy.
#
# A cost-reduction variant is included: keep 5 branches but lower the most
# expensive one (voxel25_z48 -> voxel25_z16).
#
# Uses arch_loop.measure() for multi-round retry (native crash has struck 6+ times).

param(
    [int]$Epochs = 300,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\openevolve\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_evolve5.log" }

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
        --out-dir (Join-Path $ProjectDir "data\multibranch_evolved5") `
        --log-file $Log `
        --archs `
        "voxel15_z16,voxel20_z32,voxel25_z48,voxel35_z64@gate" `
        "voxel15_z16,voxel20_z32,voxel25_z48,voxel50_z96@gate" `
        "voxel15_z16,voxel25_z48,voxel35_z64,voxel50_z96@gate" `
        "voxel20_z32,voxel25_z48,voxel35_z64,voxel50_z96@gate" `
        "voxel15_z16,voxel20_z32,voxel25_z16,voxel35_z64,voxel50_z96@gate"
    $code = $LASTEXITCODE
    Write-Host "[ev5] arch_loop exit=$code"
    Add-Content -Path $Log -Value "[ev5] arch_loop exit=$code" -Encoding UTF8
} finally {
    Pop-Location
}
