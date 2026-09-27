# Test-split acceptance of the EVOLVED champion (2026-09-25).
# ASCII-only by design (PowerShell 5.1 reads .ps1 as GBK).
#
# WHY: the evolved champion `voxel15_z16,voxel20_z32,voxel50_z96@gate` (val 0.4126)
# has never been scored on the held-out TEST rooms. Every acceptance number so far
# (the +0.0689 for the previous champion) came from a different architecture, so
# this needs its own unbiased number before it can be reported.
#
# The evolved configs are ARCH STRINGS (arch_space syntax), not registered CONFIGS --
# multibranch_exp handles them directly via _arch_spec(). Tables get the `_tval`
# suffix so they can never overwrite the search tables.
#
# Retries: native crash exit -1073740791 (0xC0000409) has struck 5 times in this
# project. Tables persist per cell, so a retry costs <=1 cell.

param(
    [int]$Epochs = 300,
    [int]$MaxRounds = 10,
    [string]$Log = "",
    # Optional override of the architecture list, so the champion can be swapped
    # from the command line without editing this file.
    [string[]]$Archs = @()
)

$ErrorActionPreference = "Continue"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PyExe = "D:\anaconda\envs\lidar310\python.exe"
if ($Log -eq "") { $Log = Join-Path $ProjectDir "_accept_evolved.log" }
$OutDir = Join-Path $ProjectDir "data\multibranch_test_evolved"

function Log([string]$m) {
    $line = "[tevo $((Get-Date).ToString('HH:mm:ss'))] $m"
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

# Baseline + the CURRENT evolved champion + the previous champion, for the
# Pareto / improvement story. Update this list whenever a new champion appears --
# acceptance numbers only ever apply to the architecture they were measured on,
# so a new champion needs its own unbiased test number.
$Names = @(
    "points",
    "voxel15_z16,voxel20_z32,voxel25_z16,voxel35_z64,voxel50_z96@gate",
    "voxel15_z16,voxel20_z32,voxel35_z64,voxel50_z96@gate",
    "ms3_mix_v20_v50_z16_z32_z96"
)
if ($Archs.Count -gt 0) { $Names = $Archs }

function Cell-Count([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return -1 }
    try {
        $raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8
        # ⚠ 不能用 '"seed"' 计数：结果表里每个 cell 的**键**是
        #   "<Area>/<room>|seed=NNNNN"，键里就含 `seed=`，
        #   但 JSON 结构是 cells -> {key -> {valroom -> miou}}，
        #   所以直接数 "|seed=" 更贴近"已完成的训练次数"。
        # 数错会导致"明明跑完了却报 INCOMPLETE"（本轮就误报了）。
        return ([regex]::Matches($raw, '\|seed=')).Count
    } catch { return -1 }
}

function All-Done {
    $ok = $true
    foreach ($n in $Names) {
        $fn = ($n -replace '\+', '_plus_') + "_k48_ep$Epochs`_tval.json"
        $f = Join-Path $OutDir $fn
        if ((Cell-Count $f) -lt 24) { $ok = $false }
    }
    return $ok
}

Push-Location $ProjectDir
try {
    for ($round = 1; $round -le $MaxRounds; $round++) {
        if (All-Done) { Log "all test tables complete. done."; break }
        $todo = @()
        foreach ($n in $Names) {
            $fn = ($n -replace '\+', '_plus_') + "_k48_ep$Epochs`_tval.json"
            $f = Join-Path $OutDir $fn
            if ((Cell-Count $f) -lt 24) { $todo += $n }
        }
        Log "round $round/$MaxRounds : pending = $($todo.Count)"
        foreach ($t in $todo) { Log "    $t" }

        $sw = [Diagnostics.Stopwatch]::StartNew()
        # NOTE: arch strings contain '@' and ',' -- quoting them in the arg list is
        # fine, but we must NOT rely on PowerShell expanding them.
        $argList = @("-W", "ignore", "-m", "nas_ptseg.multibranch_exp",
                     "--k", "48", "--epochs", "$Epochs",
                     "--seeds", "12345", "23456",
                     "--n-train", "12", "--n-val", "12", "--val-split", "test",
                     "--only") + $todo + @("--out-dir", $OutDir, "--log-file", $Log)
        & $PyExe @argList
        $code = $LASTEXITCODE
        $sw.Stop()
        Log "round $round exit=$code wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
        if ($code -ne 0) { Log "  (native crash code $code -- resuming from last persisted cell)" }
    }
    if (All-Done) { Log "final: complete" } else { Log "final: INCOMPLETE after $MaxRounds rounds" }
} finally {
    Pop-Location
}
