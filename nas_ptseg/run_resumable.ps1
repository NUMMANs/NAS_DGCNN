param(
    [Parameter(Mandatory = $true)] [string]$Module,
    [string[]]$CliArgs = @(),
    [Parameter(Mandatory = $true)] [string]$ExpectTable,
    [int]$MaxRounds = 8,
    [string]$Log = ""
)

# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# Retry wrapper for long table builds.
#
# Why it exists: the training process occasionally dies with a NATIVE crash
#     exit -1073740791 = 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN)
# with no Python traceback. It is intermittent -- the same command re-run
# immediately completes normally (verified 2026-09-24), and 4 full tables did
# finish in one go. It is NOT caused by thread env vars (OMP/MKL/OPENBLAS are
# unset here; that was a different, earlier failure mode).
#
# Table builds write the JSON after EVERY cell, so a crash costs at most one
# cell. Retrying is therefore both safe and cheap -- far cheaper than hunting an
# intermittent native bug. This wrapper retries until the table reports complete.

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_resumable.log" }

function Log([string]$m) {
    $line = "[resumable $((Get-Date).ToString('HH:mm:ss'))] $m"
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

Push-Location $ProjectDir
try {
    for ($round = 1; $round -le $MaxRounds; $round++) {
        Log "round $round/$MaxRounds : $Module $($CliArgs -join ' ')"
        $sw = [Diagnostics.Stopwatch]::StartNew()
        & $PyExe -m $Module @CliArgs --log-file $Log
        $code = $LASTEXITCODE
        $sw.Stop()
        Log "round $round exit=$code wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"

        $state = & $PyExe -c @"
import json, sys
p = r'$ExpectTable'
try:
    d = json.load(open(p, encoding='utf-8'))
except Exception as e:
    print('MISSING', e); sys.exit(0)
if 'grid' in d:
    need = len(d['grid']) * len(d['train_rooms']) * len(d['seeds'])
elif 'ks' in d:
    need = len(d['ks']) * len(d['train_rooms']) * len(d['seeds'])
else:
    print('UNKNOWN'); sys.exit(0)
have = len(d.get('cells', {}))
print('CELLS %d %d' % (have, need))
"@
        Log "  table state: $state"
        if ($state -match "CELLS (\d+) (\d+)") {
            if ([int]$Matches[1] -ge [int]$Matches[2] -and [int]$Matches[2] -gt 0) {
                Log "table complete. done."
                break
            }
        }
        if ($code -eq 0) {
            Log "exit 0 but table still incomplete; re-running once more"
        } else {
            Log "native crash (code $code) -- resuming from last persisted cell"
        }
    }
    Log "resumable wrapper finished"
} finally {
    Pop-Location
}
