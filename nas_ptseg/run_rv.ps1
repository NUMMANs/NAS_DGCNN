# Range-view round: the THIRD representation family (2026-09-25).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# Why: two representation families are already measured --
#   points (Cartesian + kNN dynamic graph)      baseline val 0.3370
#   voxel  (regular 3D grid, z as spatial axis) best val 0.4110
# range-view is the third mainstream family and has a DIFFERENT inductive bias:
#   spherical coordinates -> encodes azimuth/elevation (scanner viewpoint),
#   resolution decays with distance, and it uses cheap 2D convs.
# Adding it makes the "representation search" story complete and tests whether
# combining all three families beats the best two.
#
# NOTE on indoor adaptation: standard range-view assumes a rotating LiDAR.
# S3DIS is an indoor scan, so RangeViewBranch projects about the ROOM CENTRE
# (azimuth around the vertical axis, elevation from the horizontal plane),
# which keeps viewpoint structure + distance decay without assuming a spinner.
#
# Work list INSIDE the script: -File flattens arrays into positional args.
# Retries: intermittent native crash exit -1073740791 (0xC0000409); tables
# persist per cell so a retry costs <=1 cell.

param(
    [int]$Epochs = 300,
    [int]$MaxRounds = 15,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_rv.log" }
$OutDir = Join-Path $ProjectDir "data\multibranch_rv"

function Log([string]$m) {
    $line = "[rv $((Get-Date).ToString('HH:mm:ss'))] $m"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

$env:CUDA_VISIBLE_DEVICES = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

$Names = @(
    "points",                        # baseline
    "ms4_mix_v20_v50_z16_z32_z64_z96",   # current champion (val +0.0739)
    "ms3_mix_v20_v50_z16_z32_z96",       # cheaper 3-scale mixed (+0.0720)
    "rv32", "rv64", "rv128",         # range-view alone, resolution sweep
    "v20+rv64",                      # two families
    "v20v50+rv64",                   # three families (voxel fine + coarse + range)
    "pts+rv64",                      # points + range-view
    "v20_v50_rv64_late"              # fusion recheck with 3 branches
)

$done = @{}
foreach ($n in $Names) { $done[$n] = $false }

function Refresh-Done {
    $allOk = $true
    foreach ($n in $Names) {
        $fn = ($n -replace '\+', '_plus_') + "_k48_ep$Epochs.json"
        $f = Join-Path $OutDir $fn
        if (-not (Test-Path $f)) { $done[$n] = $false; $allOk = $false; continue }
        $txt = Get-Content -LiteralPath $f -Raw -Encoding UTF8
        $cnt = ([regex]::Matches($txt, 'seed=')).Count
        if ($cnt -ge 24) { $done[$n] = $true } else { $done[$n] = $false; $allOk = $false }
    }
    return $allOk
}

Push-Location $ProjectDir
try {
    for ($round = 1; $round -le $MaxRounds; $round++) {
        if (Refresh-Done) { Log "all tables complete. done."; break }
        $todo = @()
        foreach ($n in $Names) { if (-not $done[$n]) { $todo += $n } }
        Log "round $round/$MaxRounds : pending = $($todo -join ',')"

        $sw = [Diagnostics.Stopwatch]::StartNew()
        & $PyExe -W ignore -m nas_ptseg.multibranch_exp --k 48 --epochs $Epochs `
            --seeds 12345 23456 --n-train 12 --n-val 12 `
            --only @todo `
            --out-dir $OutDir --log-file $Log
        $code = $LASTEXITCODE
        $sw.Stop()
        Log "round $round exit=$code wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
        if ($code -ne 0) { Log "  (native crash code $code -- resuming from last persisted cell)" }
    }
    Refresh-Done | Out-Null
    Log "final: $((($Names | Where-Object { $done[$_] }).Count))/$($Names.Count) complete"
} finally {
    Pop-Location
}
