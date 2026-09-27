# Round 3: sweep the COARSE END of the champion (2026-09-25).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# WHY this axis: the evolved champion is
#     voxel15_z16 (fine) + voxel20_z32 (mid) + voxel50_z96 (coarse)   val 0.4126
# Round 2 already optimised the FINE end (z16 beats z32/z48/z64). The coarse end
# (voxel 0.50 m, z96) has never been systematically swept -- it was inherited from
# the hand-designed config and never questioned.
#
# Also tests whether a 4th branch (a full fine-to-coarse pyramid) still helps,
# i.e. whether the "number of scale branches" axis is exhausted.
#
# Uses arch_loop.measure() so we get multi-round retry -- a native crash
# (exit -1073740791 / 0xC0000409) has struck 6 times in this project and must
# never cost a whole round.

param(
    [int]$Epochs = 300,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\openevolve\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_evolve3.log" }

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
        --out-dir (Join-Path $ProjectDir "data\multibranch_evolved3") `
        --log-file $Log `
        --archs `
        "voxel15_z16,voxel20_z32,voxel35_z96@gate" `
        "voxel15_z16,voxel20_z32,voxel25_z96@gate" `
        "voxel15_z16,voxel20_z32,voxel50_z64@gate" `
        "voxel15_z16,voxel20_z32,voxel50_z128@gate" `
        "voxel15_z16,voxel20_z32,voxel35_z64,voxel50_z96@gate"
    $code = $LASTEXITCODE
    Write-Host "[ev3] arch_loop exit=$code"
    Add-Content -Path $Log -Value "[ev3] arch_loop exit=$code" -Encoding UTF8
} finally {
    Pop-Location
}
