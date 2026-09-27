# Step 2: unbiased TEST-split acceptance of the winners (2026-09-25).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# WHY THIS EXISTS: every number produced so far -- including the champion
# ms3_v20_z32_z64_z96 = 0.4062 (+0.0691) -- was measured on the VAL split.
# A result that is *selected* on val and *reported* on val is optimistic.
# This run scores the shortlisted architectures on the held-out TEST rooms,
# once, at the end. Nothing here may feed back into architecture selection.
#
# Tables get a "_tval" suffix so they can never overwrite the search tables.
#
# Retries: intermittent native crash exit -1073740791 (0xC0000409). Tables persist
# per cell, so a retry costs <=1 cell.

param(
    [int]$Epochs = 300,
    [int]$MaxRounds = 12,
    [string]$Log = ""
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_accept_t.log" }
$OutDir = Join-Path $ProjectDir "data\multibranch_test"

function Log([string]$m) {
    $line = "[tacc $((Get-Date).ToString('HH:mm:ss'))] $m"
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

# Shortlist: the baseline reference plus the accuracy leader and the two
# cost-efficiency sweet spots. Kept small on purpose -- test is touched once.
$Names = @(
    "points",                       # baseline reference
    "ms3_v20_z32_z64_z96",          # accuracy leader (val +0.0691)
    "ms_mix_v20_v50_z32_z96",       # cost sweet spot (val +0.0667, 15.0 ms/step)
    "ms_v20_z16_z96"                # 2-scale, pure-integer z ratio
)

$done = @{}
foreach ($n in $Names) { $done[$n] = $false }

function Refresh-Done {
    $allOk = $true
    foreach ($n in $Names) {
        $fn = ($n -replace '\+', '_plus_') + "_k48_ep$Epochs`_tval.json"
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
        if (Refresh-Done) { Log "all test tables complete. done."; break }
        $todo = @()
        foreach ($n in $Names) { if (-not $done[$n]) { $todo += $n } }
        Log "round $round/$MaxRounds : pending = $($todo -join ',')"

        $sw = [Diagnostics.Stopwatch]::StartNew()
        & $PyExe -W ignore -m nas_ptseg.multibranch_exp --k 48 --epochs $Epochs `
            --seeds 12345 23456 --n-train 12 --n-val 12 --val-split test `
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
