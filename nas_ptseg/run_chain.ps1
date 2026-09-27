param(
    [int]$ProbeK = 48,
    [int]$ProbeEpochs = 300,
    [int]$ProbeNTrain = 6,
    [int]$ProbeNVal = 6,
    # Only the CAPACITY axes are probed here. agg / graph_space are covered with
    # better statistics by arch_table (12 train x 12 val rooms, 2 budgets), so
    # probing them a second time would just burn GPU time for nothing.
    [string]$ProbeConfigs = "base_f64_l3_max_feat feat32 feat128 layers2 layers4",
    [int]$CheapEpochs = 50,
    [int]$FullEpochs = 300,
    [int]$PollSec = 20,
    [string]$Log = ""
)

# ASCII-only by design: PowerShell 5.1 decodes UTF-8 .ps1 as GBK and breaks on CJK.
# Single serial chain on GPU 1. Two independent "wait for idle" runners would both
# fire at the same moment and share the card -- that is exactly what this avoids.

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_chain.log" }

function Log([string]$m) {
    $line = "[chain $((Get-Date).ToString('HH:mm:ss'))] $m"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

function Wait-Idle([string]$why) {
    Log "waiting for GPU jobs to clear before: $why"
    $waited = 0
    while ($true) {
        $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                Where-Object { $_.CommandLine -like "*nas_ptseg*" }
        if (-not $busy) { break }
        Start-Sleep -Seconds $PollSec
        $waited += $PollSec
        if ($waited % 300 -eq 0) { Log "  still busy after ${waited}s" }
    }
    Log "  clear (waited ${waited}s)"
}

function Run-Step([string]$name, [string[]]$CliArgs) {
    Log "=== $name ==="
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $PyExe @CliArgs
    $code = $LASTEXITCODE
    $sw.Stop()
    Log "$name exit=$code  wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
    return $code
}

$env:CUDA_VISIBLE_DEVICES = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

Push-Location $ProjectDir
try {
    Wait-Idle "arch probe"
    $probeCfg = $ProbeConfigs.Split(" ") | Where-Object { $_ -ne "" }
    Run-Step "arch_probe" (@("-m", "nas_ptseg.arch_probe",
        "--k", "$ProbeK", "--epochs", "$ProbeEpochs",
        "--n-train", "$ProbeNTrain", "--n-val", "$ProbeNVal",
        "--configs") + $probeCfg + @("--log-file", $Log)) | Out-Null

    # Architecture table at the cheap budget first (fast), then the full budget.
    # Both are needed: the cheap one is only usable as a proxy if its RANKING of
    # architectures matches the full one -- that is exactly what compare-with checks.
    Wait-Idle "arch table @$CheapEpochs"
    Run-Step "arch_table_cheap" @("-m", "nas_ptseg.arch_table",
        "--axes-on", "k", "agg", "graph_space",
        "--epochs", "$CheapEpochs", "--log-file", $Log) | Out-Null

    Wait-Idle "arch table @$FullEpochs"
    Run-Step "arch_table_full" @("-m", "nas_ptseg.arch_table",
        "--axes-on", "k", "agg", "graph_space",
        "--epochs", "$FullEpochs", "--compare-with", "$CheapEpochs",
        "--log-file", $Log) | Out-Null

    Log "chain complete"
} finally {
    Pop-Location
}
