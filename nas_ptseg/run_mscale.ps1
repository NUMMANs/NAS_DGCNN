# Multi-scale z experiment, self-contained (no array parameter passing).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# Retries the whole run until every expected table is complete.
# Why: this workload hits an INTERMITTENT native crash
#     exit -1073740791 = 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN), no Python traceback.
# It struck at the first measure_cost() call on 2026-09-24 22:00. The same command
# re-run completes normally, and tables persist after EVERY cell, so retrying costs
# at most one cell -- far cheaper than chasing an intermittent native bug.

param(
    [int]$Epochs = 300,
    [int]$MaxRounds = 12,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_mscale_self.log" }
$OutDir = Join-Path $ProjectDir "data\multibranch_ms"

function Log([string]$m) {
    $line = "[ms $((Get-Date).ToString('HH:mm:ss'))] $m"
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

# The eight configs under test. NOTE: stored as a plain list, never as a nested
# array passed through -File (that flattens and binds positionally -> arg error).
$Names = @("points", "voxel25_z64", "ms_z32_z96", "ms_z32_z96_gate",
           "ms_z32_z96_cat", "ms_z64_z128", "ms_z16_z64", "ms3_z32_z64_z128")

$done = @{}
foreach ($n in $Names) { $done[$n] = $false }

function Refresh-Done {
    $allOk = $true
    foreach ($n in $Names) {
        $files = Get-ChildItem -Path $OutDir -Filter "$($n -replace '\+','_plus_')_k48_ep$Epochs.json" -ErrorAction SilentlyContinue
        if (-not $files) { $done[$n] = $false; $allOk = $false; continue }
        $txt = Get-Content -LiteralPath $files[0].FullName -Raw -Encoding UTF8
        # 24 cells expected (12 train rooms x 2 seeds)
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
