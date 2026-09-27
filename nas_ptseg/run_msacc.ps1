# Multi-scale ACCELERATION round: more scales (4, 5) + scale-pair choices.
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# Why a self-contained script (not run_resumable.ps1 with -CliArgs): passing an
# array through `-File` flattens it into positional args and fails with
# "cannot find a positional parameter that accepts argument '48'".
# Keep the work list INSIDE the script.
#
# Why retries: this workload intermittently dies with a NATIVE crash
#   exit -1073740791 = 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN), no traceback.
# Tables persist after EVERY cell, so a retry costs at most one cell.

param(
    [int]$Epochs = 300,
    [int]$MaxRounds = 15,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_msacc.log" }
$OutDir = Join-Path $ProjectDir "data\multibranch_msacc"

function Log([string]$m) {
    $line = "[msacc $((Get-Date).ToString('HH:mm:ss'))] $m"
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

# Configs under test. `points` is the baseline; `ms3_z32_z64_z128` is the
# reigning champion from the previous round (~0.4013 at 3 scales).
$Names = @(
    "points",
    "ms3_z32_z64_z128",          # reigning champion (3 scales)
    "ms3_cat", "ms3_late",       # same 3 scales, other fusions
    "ms4_z16_z32_z64_z128",      # 4 scales
    "ms4_z32_z64_z96_z128",      # 4 scales (denser mid range)
    "ms5_z16_z32_z64_z128_z192", # 5 scales
    "ms_z16_z192", "ms_z16_z96", "ms_z48_z128",  # scale-pair choices
    "ms_z32_z96_v35"             # mixed voxel size + scales
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
