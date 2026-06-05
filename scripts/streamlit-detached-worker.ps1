param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [int]$Port = 8501,
    [int]$TimeoutSeconds = 45,
    [string]$LogDirectory = ".codex-log"
)

$ErrorActionPreference = "Stop"

$repoRootPath = Resolve-Path $RepoRoot
$logRoot = Join-Path $repoRootPath $LogDirectory
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$stdoutLog = Join-Path $logRoot "streamlit-detached.out.log"
$stderrLog = Join-Path $logRoot "streamlit-detached.err.log"
$pidFile = Join-Path $logRoot "streamlit-detached.pid.txt"

function Get-PortListener {
    param([int]$Port)

    if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($connection) {
            return [pscustomobject]@{
                OwningProcess = $connection.OwningProcess
                Source = "Get-NetTCPConnection"
            }
        }
    }

    $listenerLine = netstat -ano |
        Select-String -Pattern (":$Port\s+.*LISTENING") |
        Select-Object -First 1
    if (-not $listenerLine) {
        return $null
    }

    $parts = ($listenerLine.Line -split "\s+") | Where-Object { $_ }
    $parsedPid = 0
    if ([int]::TryParse($parts[-1], [ref]$parsedPid)) {
        return [pscustomobject]@{
            OwningProcess = $parsedPid
            Source = "netstat"
        }
    }

    return $null
}

$existing = Get-PortListener -Port $Port
if ($existing) {
    "Reusing Streamlit listener on port $Port owned by PID $($existing.OwningProcess) via $($existing.Source)." |
        Set-Content -Path $pidFile
    exit 0
}

$poetry = (Get-Command poetry).Source
$arguments = @(
    "run",
    "streamlit",
    "run",
    "gllm/code_generator_streamlit_reasoning_langchain_langgraph.py",
    "--server.port",
    "$Port",
    "--server.headless",
    "true",
    "--browser.gatherUsageStats",
    "false"
)

$process = Start-Process `
    -FilePath $poetry `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRootPath `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

"Started Streamlit launcher PID $($process.Id) for port $Port." | Set-Content -Path $pidFile

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    $listener = Get-PortListener -Port $Port
    if ($listener) {
        Add-Content -Path $pidFile -Value "Listening PID $($listener.OwningProcess) via $($listener.Source)."
        exit 0
    }

    if ($process.HasExited) {
        "Streamlit launcher exited before port $Port was ready. See $stderrLog" |
            Add-Content -Path $pidFile
        exit 1
    }

    Start-Sleep -Seconds 1
} while ((Get-Date) -lt $deadline)

"Timed out waiting for Streamlit on port $Port. Launcher PID $($process.Id). See $stderrLog" |
    Add-Content -Path $pidFile
exit 1
