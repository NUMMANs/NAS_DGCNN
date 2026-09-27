# Round 4: push the BRANCH-COUNT / scale-coverage axis (2026-09-25).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# WHY: round 3 produced a clean negative + a clean positive.
#   NEGATIVE: perturbing the coarse-end PARAMETERS did nothing
#             (v50->v35/v25, z96->z64/z128 all <= -0.0001, some much worse).
#   POSITIVE: adding a 4TH BRANCH won (+0.0021) -> 0.4147, a new champion.
# => the remaining gain is in "how many scale branches", not "what values they take".
# Round 4 therefore tests 5-branch pyramids and the 4-branch neighbourhood.
#
# Uses arch_loop.measure() for multi-round retry (native crash has struck 6 times).

param(
    [int]$Epochs = 300,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\openevolve\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_evolve4.log" }

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
        --out-dir (Join-Path $ProjectDir "data\multibranch_evolved4") `
        --log-file $Log `
        --archs `
        "voxel15_z16,voxel20_z32,voxel25_z48,voxel35_z64,voxel50_z96@gate" `
        "voxel12_z16,voxel15_z32,voxel20_z48,voxel35_z64,voxel50_z96@gate" `
        "voxel15_z16,voxel20_z32,voxel35_z64,voxel50_z128@gate" `
        "voxel15_z16,voxel20_z32,voxel35_z48,voxel50_z96@gate" `
        "voxel15_z32,voxel20_z32,voxel35_z64,voxel50_z96@gate"
    $code = $LASTEXITCODE
    Write-Host "[ev4] arch_loop exit=$code"
    Add-Content -Path $Log -Value "[ev4] arch_loop exit=$code" -Encoding UTF8
} finally {
    Pop-Location
}
