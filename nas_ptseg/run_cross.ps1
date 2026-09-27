# Cross-product round: multi-scale x mixed-voxel-size (2026-09-25).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# Two axes are each proven effective:
#   (1) z multi-scale : 1 scale 0.3885 -> 5 scales 0.4046
#   (2) voxel size    : inverted-U, optimum 0.20 m (v50 0.3685 < v15 0.3879 < v20 0.3902)
#                        and MIXING sizes in one net is the cost sweet spot
#                        (ms_mix_v20_v50 = 0.4037 @ 15.0 ms vs 3-scale 0.4062 @ 56.0 ms)
# Their CROSS PRODUCT is not exhausted: 3/4-scale mixed, more scale pairs, and a
# gate-vs-cat-vs-late recheck at the larger config were never measured.
#
# Work list INSIDE the script: passing an array via -File flattens it into
# positional args and fails.
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
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_cross.log" }
$OutDir = Join-Path $ProjectDir "data\multibranch_cross"

function Log([string]$m) {
    $line = "[cross $((Get-Date).ToString('HH:mm:ss'))] $m"
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
    "points",                                  # baseline
    "ms3_v20_z32_z64_z96",                     # accuracy leader (test +0.0689)
    "ms_mix_v20_v50_z32_z96",                  # cost sweet spot (test +0.0642, 16.8 ms)
    "ms3_mix_v20_v50_z16_z32_z96",             # 3-scale mixed
    "ms3_mix_v20_v50_z32_z64_z96",             # 3-scale mixed (other pair)
    "ms4_mix_v20_v50_z16_z32_z64_z96",         # 4-scale mixed
    "ms_mix_v20_v35_z32_z96",                  # mixed w/ v35 coarse end
    "ms_mix_v15_v20_z16_z96",                  # fine-fine mix
    "ms_mix_v20_v50_z16_z64",                  # scale-pair scan
    "ms_mix_v20_v50_z16_z128",
    "ms_mix_v20_v50_z32_z128",
    "ms_mix_v20_v50_cat",                      # fusion recheck at larger config
    "ms_mix_v20_v50_late",
    "ms_mix_v35_v50_z32_z96"                   # cheapest direction probe
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
