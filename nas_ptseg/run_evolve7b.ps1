# Round 7b: FEATURE WIDTH axis, via arch_loop for retry protection (2026-09-26).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# WHY the relaunch: round 7 was launched straight through multibranch_exp, which has
# NO retry protection -- the process died after cell 1 (native crash, exit 1, no
# traceback) and wrote nothing. Round 7b goes through arch_loop.measure(), which
# retries up to 10 rounds and only re-runs the cells that are actually missing.
#
# WHY the list changed:
#   The first probe already produced a decisive cost fact --
#   the 5-branch champion with feat=128 measures 131.6 ms/step, which is SLOWER than
#   the points baseline (104 ms). That config can never be Pareto-optimal, so it is
#   dropped rather than re-run.
#   What remains is the actually informative part of the axis:
#     - feat=32 on the champion  -> can we get CHEAPER without losing accuracy?
#     - feat=128 on the 3- and 4-branch (cheaper) configs
#       -> is buying capacity cheaper here than in the points family?
#
# NOTE for the cost model: `arch_space.estimate_cost()` does NOT model `feat`
# (it underestimates feat=128 badly: predicted 46.7 ms vs measured 131.6 ms).
# Treat synth-cost as a lower bound whenever feat != 64.

param(
    [int]$Epochs = 300,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\openevolve\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_evolve7b.log" }

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
        --out-dir (Join-Path $ProjectDir "data\multibranch_evolved7") `
        --log-file $Log `
        --archs `
        "voxel15_z16,voxel20_z32,voxel25_z16,voxel35_z64,voxel50_z96@gate|feat=32" `
        "voxel15_z16,voxel20_z32,voxel50_z96@gate|feat=128" `
        "voxel15_z16,voxel20_z32,voxel35_z64,voxel50_z96@gate|feat=128"
    $code = $LASTEXITCODE
    Write-Host "[ev7b] arch_loop exit=$code"
    Add-Content -Path $Log -Value "[ev7b] arch_loop exit=$code" -Encoding UTF8
} finally {
    Pop-Location
}
