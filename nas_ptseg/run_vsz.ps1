# Voxel-size x scale JOINT search (2026-09-25).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# Context: the multi-scale axis is proven (1 scale 0.3885 -> 5 scales 0.4046),
# but almost every config so far used voxel_size=0.25 m. Only a single v35
# variant was ever tried. This round separates the two axes:
#   (a) single-scale, sweeping voxel size at fixed z=64  -> isolates voxel size
#   (b) multi-scale, sweeping voxel size                 -> joint effect
#   (c) MIXED voxel sizes inside one net                 -> is voxel size also a scale axis?
#
# Work list is INSIDE the script: passing an array via -File flattens it into
# positional args and fails ("cannot find a positional parameter accepting '48'").
#
# Retries: this workload intermittently dies with exit -1073740791 (0xC0000409,
# native, no traceback). Tables persist after every cell -> a retry costs <=1 cell.

param(
    [int]$Epochs = 300,
    [int]$MaxRounds = 15,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_vsz.log" }
$OutDir = Join-Path $ProjectDir "data\multibranch_vsz"

function Log([string]$m) {
    $line = "[vsz $((Get-Date).ToString('HH:mm:ss'))] $m"
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
    "points",                    # baseline
    "ms_z16_z96",                # best 2-scale at 0.25 m (reference: +0.0618, 6.7 s/run)
    "v15_z64", "v20_z64", "v35_z64", "v50_z64",   # (a) isolate voxel size
    "ms_v15_z16_z96", "ms_v20_z16_z96", "ms_v35_z16_z96", "ms_v50_z16_z96",  # (b)
    "ms_mix_v15_v35_z32_z96", "ms_mix_v20_v50_z32_z96",   # (c)
    "ms3_v20_z32_z64_z96"        # 3-scale + finer voxel
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
