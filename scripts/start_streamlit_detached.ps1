param(
    [int]$Port = 8501,
    [int]$TimeoutSeconds = 45,
    [string]$LogDirectory = ".codex-log"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$logRoot = Join-Path $repoRoot $LogDirectory
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$stdoutLog = Join-Path $logRoot "streamlit-detached.out.log"
$stderrLog = Join-Path $logRoot "streamlit-detached.err.log"
$pidFile = Join-Path $logRoot "streamlit-detached.pid.txt"
$workerScript = Join-Path $PSScriptRoot "streamlit-detached-worker.ps1"

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
    Write-Host "Streamlit is already listening at http://localhost:$Port (PID $($existing.OwningProcess))."
    Write-Host "PID file: $pidFile"
    exit 0
}

function Get-PowerShellHost {
    $pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($pwsh) {
        return $pwsh.Source
    }

    $powershell = Get-Command powershell -ErrorAction SilentlyContinue
    if ($powershell) {
        return $powershell.Source
    }

    throw "Could not find pwsh or powershell to launch Streamlit worker."
}

function Start-StreamlitWorker {
    param(
        [string]$PowerShellHost,
        [string]$WorkerScript,
        [string]$RepoRoot,
        [string]$LogDirectory,
        [int]$Port,
        [int]$TimeoutSeconds
    )

    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $WorkerScript,
        "-RepoRoot",
        $RepoRoot,
        "-LogDirectory",
        $LogDirectory,
        "-Port",
        "$Port",
        "-TimeoutSeconds",
        "$TimeoutSeconds"
    )

    return Start-Process `
        -FilePath $PowerShellHost `
        -ArgumentList $arguments `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -PassThru
}

$powerShellHost = Get-PowerShellHost
$worker = Start-StreamlitWorker `
    -PowerShellHost $powerShellHost `
    -WorkerScript $workerScript `
    -RepoRoot $repoRoot `
    -LogDirectory $LogDirectory `
    -Port $Port `
    -TimeoutSeconds $TimeoutSeconds

@(
    "Worker launcher PID $($worker.Id) for Streamlit port $Port.",
    "The worker writes readiness and Streamlit PID details after startup.",
    "Logs: $stdoutLog",
    "Errors: $stderrLog"
) | Set-Content -Path $pidFile

Write-Host "Started Streamlit worker for http://localhost:$Port (worker PID $($worker.Id))."
Write-Host "Logs: $stdoutLog"
Write-Host "Errors: $stderrLog"
Write-Host "PID file: $pidFile"
exit 0
