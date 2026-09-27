# =============================================================================
# Cheap-budget ladder: how long must "cheap" training be before its ranking
# predicts the full-training ranking?
#
# Runs sequentially on the SAME GPU (GPU 1) on purpose: launching several
# training processes at once would contend for the same card, which the user
# explicitly forbade ("never squeeze out another program's resources").
#
# ASCII-only on purpose: PowerShell 5.1 decodes .ps1 as ANSI (GBK on a Chinese
# Windows) and non-ASCII comments break parsing.
# =============================================================================

param(
    [int[]]$Epochs = @(50, 100, 150),
    [int[]]$Seeds  = @(12345, 23456),
    [string]$FullTable = ""
)

$ErrorActionPreference = "Stop"

# Sanity guard.
# `powershell -File script.ps1 -Epochs 50,100,150` passes "50,100,150" as ONE string,
# which used to bind as the single int 50100150 -- that silently kicked off a table
# build at 50 million epochs. Refuse anything absurd instead.
foreach ($e in $Epochs) {
    if ($e -lt 1 -or $e -gt 2000) {
        Write-Host "[ladder] FATAL: epoch budget $e is out of range (1..2000)."
        Write-Host "[ladder] If you launched via 'powershell -File', pass the array as"
        Write-Host "[ladder]   -Epochs @(50,100,150)   or call the script with '&' instead,"
        Write-Host "[ladder] because -File binds '50,100,150' as a single string."
        exit 2
    }
}

$ProjectDir = Split-Path -Parent $PSScriptRoot
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
$Log = Join-Path $ProjectDir "_cheap_ladder.log"

function Log($m) {
    $line = "[ladder] $m"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

# Keep the whole batch pinned to physical GPU 1 (belt and braces: nas_ptseg/gpu.py
# enforces it again at import time).
$env:CUDA_VISIBLE_DEVICES = "1"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

"" | Set-Content -Path $Log -Encoding UTF8
Log "start; epochs=$($Epochs -join ','); seeds=$($Seeds -join ',')"

Push-Location $ProjectDir
try {
    foreach ($e in $Epochs) {
        Log "=== build cheap table at $e epochs ==="
        & $PyExe -m nas_ptseg.fitness --role cheap --epochs $e --seeds @Seeds `
            --log-file (Join-Path $ProjectDir "_table_cheap_ep$e.progress.log") `
            2>&1 | Out-Null
        Log "exit=$LASTEXITCODE  (cheap @ $e epochs)"

        Log "=== reliability: cheap@$e vs full ==="
        # NOTE: do not name this $args -- that is a PowerShell automatic variable
        # and assigning to it inside a script scope causes subtle breakage.
        $pyArgs = @("-m", "nas_ptseg.reliability", "--log-file", $Log)
        if ($FullTable -ne "") { $pyArgs += @("--full", $FullTable) }
        $pyArgs += @("--cheap-epochs", "$e")
        $pyArgs += @("--out", (Join-Path $ProjectDir "_reliability_cheap${e}.json"))
        & $PyExe @pyArgs 2>&1 | Out-Null
        Log "exit=$LASTEXITCODE  (reliability @ $e)"
    }
} finally {
    Pop-Location
}

Log "done"
