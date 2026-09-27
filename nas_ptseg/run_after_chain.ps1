param(
    [int]$Epochs = 300,
    [int]$PollSec = 30,
    [string]$Log = ""
)

# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# Waits for the arch chain to finish, then builds the **held-out acceptance table**:
# training rooms x TEST rooms (fitness.py --val-split test, tag "tval").
#
# Why it must exist before any claim is made: every number produced so far --
# including the search itself -- was measured on the val split. A policy chosen on
# val and reported on val is optimistic. The test table gives the unbiased number
# and it is built once, at the end, so nothing can leak into the search loop.

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_accept.log" }
$ChainLog = Join-Path $ProjectDir "_chain.log"

function Log([string]$m) {
    $line = "[accept $((Get-Date).ToString('HH:mm:ss'))] $m"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

Log "waiting for the arch chain to report 'chain complete' ..."
$waited = 0
while ($true) {
    $done = $false
    if (Test-Path $ChainLog) {
        $txt = Get-Content -LiteralPath $ChainLog -Raw -Encoding UTF8
        if ($txt -and ($txt -match "chain complete")) { $done = $true }
    }
    if ($done) { break }
    Start-Sleep -Seconds $PollSec
    $waited += $PollSec
    if ($waited % 600 -eq 0) { Log "  still waiting (${waited}s)" }
}
Log "chain finished (waited ${waited}s). building the held-out acceptance table."

$env:CUDA_VISIBLE_DEVICES = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"

Push-Location $ProjectDir
try {
    Log "=== full table, val split = test (tag tval) ==="
    & $PyExe -m nas_ptseg.fitness --role full --epochs $Epochs `
        --val-split test --log-file $Log
    Log "accept-table exit=$LASTEXITCODE"
} finally {
    Pop-Location
}
