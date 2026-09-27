param(
    [int]$K = 48,
    [int]$Epochs = 300,
    [int]$NTrain = 6,
    [int]$NVal = 6,
    [int]$PollSec = 20,
    [string]$Log = ""
)

# ASCII-only by design: PowerShell 5.1 decodes UTF-8 .ps1 as GBK and breaks on CJK.

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"

if ($Log -eq "") { $Log = Join-Path $ProjectDir "_arch_probe.log" }
if (Test-Path $Log) { Remove-Item $Log -Force }

function Log([string]$m) {
    $line = "[wait] $m"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

# Wait until no other fitness/table job is running: the ladder must finish first,
# otherwise two jobs would share GPU 1 and slow each other down.
Log "waiting for any running nas_ptseg job to finish ..."
$waited = 0
while ($true) {
    $busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -like "*nas_ptseg*" }
    if (-not $busy) { break }
    Start-Sleep -Seconds $PollSec
    $waited += $PollSec
    if ($waited % 300 -eq 0) { Log "still busy after ${waited}s" }
}
Log "GPU is free (waited ${waited}s). starting arch probe."
Log "target_n fixed at 16384; k=$K epochs=$Epochs train=$NTrain val=$NVal"

$env:CUDA_VISIBLE_DEVICES = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

Push-Location $ProjectDir
try {
    & $PyExe -m nas_ptseg.arch_probe --k $K --epochs $Epochs `
        --n-train $NTrain --n-val $NVal --log-file $Log
    Log "arch_probe exit=$LASTEXITCODE"
} finally {
    Pop-Location
}
