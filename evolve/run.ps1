# =============================================================================
# OpenEvolve launcher for the NAS x point-cloud-segmentation task.
#
# IMPORTANT: this file is deliberately ASCII-only.
# Windows PowerShell 5.1 reads .ps1 files using the system ANSI codepage (GBK on
# a Chinese Windows). Non-ASCII characters in comments get mis-decoded and can
# break parsing with confusing syntax errors. Keep all comments in English here.
#
# Usage:
#   .\run.ps1                     # full run with config.yaml
#   .\run.ps1 -DryRun             # offline preflight only, no API calls
#   .\run.ps1 -Iterations 30      # override max_iterations
#   .\run.ps1 -LogLevel DEBUG
# =============================================================================

param(
    [switch]$DryRun,
    [int]$Iterations = 0,
    [string]$LogLevel = "INFO",
    # Search space to evolve.
    #   k      -> local neighbourhood size only      (config.yaml / evaluator.py)
    #   arch   -> pooling / graph-space (+k)         (config_arch.yaml / evaluator_arch.py)
    #   pareto     -> accuracy x cost, SELECT from measured candidates
    #   pareto-gen -> accuracy x cost, and the policy may PROPOSE NEW architectures
    # The k space was measured to be exhausted (per-room oracle headroom +0.0017 mIoU,
    # below noise). The arch space has a main effect but ZERO interaction, so
    # "conditioned architecture" cannot win there.
    # 'pareto' is the current main line: geometric representation changes shift BOTH
    # accuracy and cost (measured ~25x cost spread: 11.7 .. 243 ms/step), so the
    # MAP-Elites cost dimension finally has real variation to explore.
    # 'pareto-gen' additionally lets the policy emit a brand-new branch/fuse/z_bins
    # combination; unseen combos are scored with a CONSERVATIVE estimate so only
    # genuinely promising ones can beat the measured candidates.
    [ValidateSet("k", "arch", "pareto", "pareto-gen")]
    [string]$Space = "k"
)

$ErrorActionPreference = "Stop"

$EvolveDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $EvolveDir
$PyExe = "D:\anaconda\envs\openevolve\python.exe"

if ($Space -eq "pareto-gen") {
    $ConfigPath = Join-Path $EvolveDir "config_pareto.yaml"
    $InitProg = Join-Path $EvolveDir "initial_program_pareto_gen.py"
    $Evaluator = Join-Path $EvolveDir "evaluator_pareto.py"
    $OutputDir = Join-Path $EvolveDir "openevolve_output_pareto_gen"
} elseif ($Space -eq "pareto") {
    $ConfigPath = Join-Path $EvolveDir "config_pareto.yaml"
    $InitProg = Join-Path $EvolveDir "initial_program_pareto.py"
    $Evaluator = Join-Path $EvolveDir "evaluator_pareto.py"
    $OutputDir = Join-Path $EvolveDir "openevolve_output_pareto"
} elseif ($Space -eq "arch") {
    $ConfigPath = Join-Path $EvolveDir "config_arch.yaml"
    $InitProg = Join-Path $EvolveDir "initial_program_arch.py"
    $Evaluator = Join-Path $EvolveDir "evaluator_arch.py"
    $OutputDir = Join-Path $EvolveDir "openevolve_output_arch"
} else {
    $ConfigPath = Join-Path $EvolveDir "config.yaml"
    $InitProg = Join-Path $EvolveDir "initial_program.py"
    $Evaluator = Join-Path $EvolveDir "evaluator.py"
    $OutputDir = Join-Path $EvolveDir "openevolve_output"
}
$EnvFile = Join-Path $EvolveDir ".env"
$CacheDir = Join-Path $ProjectDir "data\cache_s3dis"

function Write-Head($t) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
    Write-Host $t -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkCyan
}

function Write-Ok($t)   { Write-Host "  [OK]   $t" -ForegroundColor Green }
function Write-Warn($t) { Write-Host "  [WARN] $t" -ForegroundColor Yellow }
function Write-Err($t)  { Write-Host "  [FAIL] $t" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# 1. Environment: clear proxies
# ---------------------------------------------------------------------------
# The LLM gateway is ONLY reachable by a direct connection on this machine.
# Any inherited HTTP_PROXY / HTTPS_PROXY makes the OpenAI client route through
# a proxy that cannot reach it, and every request fails.
# This must happen before anything else touches the network.
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:NO_PROXY = "*"
# Force UTF-8 so Chinese log output is not mangled.
$env:PYTHONIOENCODING = "utf-8"
# CRITICAL: openevolve's Config.from_yaml opens config.yaml WITHOUT encoding="utf-8",
# so on a Chinese Windows it falls back to the ANSI codepage (GBK) and dies with
#     UnicodeDecodeError: 'gbk' codec can't decode byte 0x80
# on the Chinese comments in this config. PYTHONUTF8=1 turns on Python's UTF-8 mode,
# which makes the default open() encoding UTF-8 everywhere. Do not remove this line
# unless config.yaml is rewritten as pure ASCII.
$env:PYTHONUTF8 = "1"

# Make PowerShell decode child-process stdout as UTF-8 too. Without this,
# `$smoke = & $PyExe $Evaluator | Out-String` reads Python's UTF-8 bytes using the
# ANSI codepage and every Chinese log line turns into mojibake. Purely cosmetic
# (the authoritative logs are written by Python itself via --log-file), but it
# keeps the console readable.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # Non-interactive hosts may refuse this; it is only cosmetic, so ignore.
}

Write-Head "OpenEvolve launcher : NAS x point cloud segmentation"

# ---------------------------------------------------------------------------
# 2. Load secrets from .env
# ---------------------------------------------------------------------------
Write-Host "Config    : $ConfigPath"
Write-Host "Initial   : $InitProg"
Write-Host "Evaluator : $Evaluator"
Write-Host "Output    : $OutputDir"

if (Test-Path $EnvFile) {
    $loaded = 0
    # -Encoding UTF8 is REQUIRED, not cosmetic.
    # Windows PowerShell 5.1 defaults to the ANSI codepage (GBK here). Decoding a
    # UTF-8 file as GBK makes multi-byte sequences swallow the following byte; when
    # that byte is CR/LF the line break disappears and two lines merge -- which can
    # silently drop the DEEPSEEK_API_KEY line entirely ("loaded 0 variable(s)").
    # .env is also kept pure ASCII so the failure mode cannot come back.
    foreach ($line in Get-Content -LiteralPath $EnvFile -Encoding UTF8) {
        $t = $line.Trim()
        if ($t -eq "" -or $t.StartsWith("#")) { continue }
        $i = $t.IndexOf("=")
        if ($i -lt 1) { continue }
        $name = $t.Substring(0, $i).Trim()
        $value = $t.Substring($i + 1).Trim()
        # strip optional surrounding quotes
        if ($value.Length -ge 2) {
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
        $loaded++
    }
    Write-Ok "loaded $loaded variable(s) from evolve/.env"
} else {
    Write-Warn "evolve/.env not found"
    Write-Warn "create it from .env.example and put your real key in it"
}

if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    Write-Err "DEEPSEEK_API_KEY is not set"
    Write-Host "         Add this line to evolve/.env :" -ForegroundColor Yellow
    Write-Host "             DEEPSEEK_API_KEY=sk-your-real-key" -ForegroundColor Yellow
    Write-Host "         Then re-run. (.env is gitignored, the key stays local.)" -ForegroundColor Yellow
    exit 1
}
Write-Ok "DEEPSEEK_API_KEY present (length $($env:DEEPSEEK_API_KEY.Length))"

# ---------------------------------------------------------------------------
# 3. Preflight: python, cache, files
# ---------------------------------------------------------------------------
Write-Head "Preflight"

if (-not (Test-Path $PyExe)) { Write-Err "python not found: $PyExe"; exit 1 }
Write-Ok "python: $PyExe"

foreach ($f in @($ConfigPath, $InitProg, $Evaluator)) {
    if (-not (Test-Path $f)) { Write-Err "missing file: $f"; exit 1 }
}
Write-Ok "config / initial_program / evaluator all present"

$cacheNpz = Join-Path $CacheDir "s3dis_domains.npz"
$cacheMeta = Join-Path $CacheDir "meta.json"
$SplitFile = Join-Path $ProjectDir "data\room_split.json"
$CheapTable = Join-Path $ProjectDir "data\fitness_tables\cheap_ep50_s12345-23456.json"
$ArchTable = Join-Path $ProjectDir "data\arch_tables\arch_k+agg+graph_space_n6_ep50_s12345-23456.json"

# The evaluator no longer reads the old single-room density cache; it scores
# candidates by looking up a precomputed cross-room table. So the things
# that actually must exist are the room split and that table.
if (-not (Test-Path $SplitFile)) {
    Write-Err "room split not found: $SplitFile"
    Write-Host ""
    Write-Host "         Build it first:" -ForegroundColor Yellow
    Write-Host "             python -m nas_ptseg.room_split --build" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
Write-Ok "fixed room split present"

if ($Space -eq "arch") {
    if (-not (Test-Path $ArchTable)) {
        Write-Err "architecture table not found: $ArchTable"
        Write-Host ""
        Write-Host "         Build it first (needs GPU 1 + torch env):" -ForegroundColor Yellow
        Write-Host "             python -m nas_ptseg.arch_table --axes-on k agg graph_space --epochs 50" -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
    Write-Ok "architecture table present"
} elseif ($Space -eq "pareto" -or $Space -eq "pareto-gen") {
    # The pareto evaluator globs data/multibranch*/*.json and keeps only tables that
    # have a MEASURED step_sec (parameter count is NOT a valid cost proxy here:
    # voxel vs points differ 2.2x in params but 20x in wall time).
    $mbDirs = Get-ChildItem -Path (Join-Path $ProjectDir "data") -Directory -Filter "multibranch*" -ErrorAction SilentlyContinue
    $measured = 0
    foreach ($d in $mbDirs) {
        foreach ($f in Get-ChildItem -Path $d.FullName -Filter "*.json" -ErrorAction SilentlyContinue) {
            if ($f.Name -like "summary*") { continue }
            try {
                $raw = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
                if ($raw -match '"step_sec"\s*:\s*[0-9.]+') { $measured++ }
            } catch { }
        }
    }
    if ($measured -lt 4) {
        Write-Err "only $measured table(s) with measured step_sec found"
        Write-Host ""
        Write-Host "         Build candidates first (needs GPU 1 + torch env):" -ForegroundColor Yellow
        Write-Host "             python -m nas_ptseg.multibranch_exp --n-train 12 --n-val 12" -ForegroundColor Yellow
        Write-Host "                 --only points voxel25_z64 ms3_v20_z32_z64_z96 ms4_mix_v20_v50_z16_z32_z64_z96" -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
    Write-Ok "multi-branch tables with measured cost: $measured"
} else {
    if (-not (Test-Path $CheapTable)) {
        Write-Err "cheap fitness table not found: $CheapTable"
        Write-Host ""
        Write-Host "         Build it first (needs GPU 1 + torch env):" -ForegroundColor Yellow
        Write-Host "             python -m nas_ptseg.fitness --role cheap --epochs 50 --seeds 12345 23456" -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
    Write-Ok "cheap fitness table present (50 epochs: measured Spearman vs full = +0.943)"
}

# The legacy density cache is only needed by the old single-room tooling; warn
# but do not block, since the new evaluator does not touch it.
if (-not (Test-Path $cacheNpz)) {
    Write-Warn "legacy density-domain cache missing (only old single-room tools need it)"
}

# ---------------------------------------------------------------------------
# 4. Verify the evaluator actually runs (offline, no API calls)
# ---------------------------------------------------------------------------
Write-Head "Evaluator smoke test (offline)"

Push-Location $ProjectDir
try {
    # The evaluator prints its candidate/base summary to STDERR on purpose (it must
    # not pollute the metrics JSON on stdout). With $ErrorActionPreference = "Stop",
    # PowerShell would turn that stderr output into a TERMINATING error and abort the
    # launcher even though the evaluator succeeded. Temporarily relax to "Continue"
    # around this call; the real gate is the exit code checked below.
    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $smoke = & $PyExe $Evaluator 2>&1 | Out-String
    $smokeCode = $LASTEXITCODE
    $ErrorActionPreference = $savedEAP
} finally {
    Pop-Location
}

if ($smokeCode -ne 0) {
    Write-Err "evaluator smoke test failed (exit $smokeCode)"
    Write-Host $smoke
    exit 1
}
Write-Ok "evaluator runs"

# Print the key metrics block so the baseline is visible before spending tokens.
$metricLines = ($smoke -split "`r?`n") | Where-Object { $_ -match "combined_score|mean_room_miou|worst_room_miou|n_distinct_cfgs|cfg_switch_rate|k_spread|log10_cost|cost_ms|speedup_vs_base|delta_vs_base|n_candidates" }
foreach ($l in $metricLines) { Write-Host "  $l" -ForegroundColor Gray }

if ($DryRun) {
    Write-Head "DryRun complete"
    Write-Host "  All preflight checks passed. Nothing was sent to the API." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Reminder: this round verifies whether the CHEAP fitness ranking" -ForegroundColor Yellow
    Write-Host "  predicts the FULL-training ranking. Check that with:" -ForegroundColor Yellow
    Write-Host "      python -m nas_ptseg.reliability --cheap-epochs 25" -ForegroundColor Yellow
    Write-Host "  Then launch a small search:" -ForegroundColor Yellow
    Write-Host "      .\run.ps1 -Iterations 20" -ForegroundColor Yellow
    exit 0
}

# ---------------------------------------------------------------------------
# 5. Show which models the endpoint actually serves
# ---------------------------------------------------------------------------
# Model names change between provider releases; a stale name in config.yaml
# would fail on the very first LLM call. Ask the endpoint instead of guessing.
# The base URL is read from config.yaml so this stays correct when the gateway
# changes (it is no longer api.deepseek.com).
$ApiBase = ""
$cfgRaw = Get-Content $ConfigPath -Raw
$m = [regex]::Match($cfgRaw, 'api_base:\s*"([^"]+)"')
if ($m.Success) { $ApiBase = $m.Groups[1].Value.TrimEnd("/") }

if ([string]::IsNullOrWhiteSpace($ApiBase)) {
    Write-Warn "could not read api_base from config.yaml; skipping model list"
} else {
    Write-Head "Available models on $ApiBase"

    try {
        $resp = Invoke-RestMethod -Uri "$ApiBase/models" `
            -Headers @{ Authorization = "Bearer $($env:DEEPSEEK_API_KEY)" } `
            -TimeoutSec 30 -Proxy $null
        $names = @()
        foreach ($mm in $resp.data) { $names += $mm.id }
        if ($names.Count -gt 0) {
            foreach ($n in $names) { Write-Host "  - $n" -ForegroundColor Gray }
        } else {
            Write-Warn "endpoint returned an empty model list"
        }
        foreach ($n in $names) {
            if ($cfgRaw -notmatch [regex]::Escape($n)) {
                Write-Warn "config.yaml does not mention model '$n' -- check the names there"
            }
        }
        if ($names.Count -gt 0 -and $names.Count -lt 2) {
            Write-Warn "endpoint serves only $($names.Count) model; the ensemble diversity is limited"
        }
    } catch {
        Write-Warn "could not list models: $($_.Exception.Message)"
        Write-Warn "continuing anyway; a wrong model name will fail on the first LLM call"
    }
}

# ---------------------------------------------------------------------------
# 6. Launch
# ---------------------------------------------------------------------------
Write-Head "Launching evolution"

# NOTE: do not name this variable $args -- that is a PowerShell automatic
# variable and assigning to it inside a script scope causes subtle breakage.
$cliArgs = @($InitProg, $Evaluator, "--config", $ConfigPath, "--output", $OutputDir, "--log-level", $LogLevel)
if ($Iterations -gt 0) {
    $cliArgs += @("--iterations", "$Iterations")
}

$CliExe = Join-Path (Split-Path -Parent $PyExe) "Scripts\openevolve-run.exe"
if (Test-Path $CliExe) {
    Write-Host "  openevolve-run $($cliArgs -join ' ')" -ForegroundColor Gray
} else {
    $CliExe = $PyExe
    $cliArgs = @("-m", "openevolve.cli") + $cliArgs
    Write-Host "  python -m openevolve.cli $($cliArgs -join ' ')" -ForegroundColor Gray
}
Write-Host ""
Write-Host "  Notes:" -ForegroundColor DarkGray
Write-Host "    - Proxies are cleared for this process (DeepSeek needs a direct connection)." -ForegroundColor DarkGray
Write-Host "    - Progress and the best program land in: $OutputDir" -ForegroundColor DarkGray
Write-Host ""

Push-Location $ProjectDir
try {
    & $CliExe @cliArgs
    $runCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Head "Finished (exit $runCode)"
Write-Host "  Results: $OutputDir" -ForegroundColor Gray
Write-Host "  Best program:  $OutputDir\best\best_program.py" -ForegroundColor Gray
Write-Host "  Best info:     $OutputDir\best\best_program_info.json" -ForegroundColor Gray

exit $runCode
