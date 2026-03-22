[CmdletBinding()]
param(
    [string]$VenvPath = ".deepgaze-py312",
    [switch]$ForceRecreate,
    [switch]$RunFullValidation,
    [switch]$SkipVcRuntime,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param([string]$Message)
    Write-Host "[DeepGaze Setup] $Message" -ForegroundColor Cyan
}

function Get-RepoRoot {
    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}

function Get-AbsolutePath {
    param(
        [string]$BasePath,
        [string]$CandidatePath
    )

    if ([System.IO.Path]::IsPathRooted($CandidatePath)) {
        return [System.IO.Path]::GetFullPath($CandidatePath)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $CandidatePath))
}

function Find-Python312 {
    $pathFromLauncher = $null
    try {
        $pathFromLauncher = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1)
    } catch {
        $pathFromLauncher = $null
    }

    if ($pathFromLauncher) {
        return $pathFromLauncher.ToString().Trim()
    }

    try {
        $pathFromPython = (& python -c "import sys; print(sys.executable if sys.version_info[:2] == (3, 12) else '')" 2>$null | Select-Object -First 1)
    } catch {
        $pathFromPython = $null
    }

    if ($pathFromPython) {
        $candidate = $pathFromPython.ToString().Trim()
        if ($candidate) {
            return $candidate
        }
    }

    throw "Python 3.12 was not found. Install Python 3.12 and make sure `py -3.12` works."
}

function Get-VcRuntimeVersion {
    try {
        $runtime = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
        if ($runtime.Version) {
            return [version]$runtime.Version.ToString().TrimStart("v")
        }
    } catch {
        return $null
    }

    return $null
}

function Ensure-VcRuntime {
    param([version]$MinimumVersion = [version]"14.50.35719.0")

    $currentVersion = Get-VcRuntimeVersion
    if ($currentVersion -and $currentVersion -ge $MinimumVersion) {
        Write-Step "VC++ x64 runtime already satisfies the minimum version ($currentVersion)."
        return
    }

    $installerDir = Join-Path ([System.IO.Path]::GetTempPath()) "gaze-toolkit-deepgaze"
    $installerPath = Join-Path $installerDir "vc_redist.x64.exe"
    New-Item -ItemType Directory -Force -Path $installerDir | Out-Null

    Write-Step "Installing Microsoft Visual C++ x64 runtime $MinimumVersion or newer."
    Invoke-WebRequest -Uri "https://aka.ms/vc14/vc_redist.x64.exe" -OutFile $installerPath
    $process = Start-Process -FilePath $installerPath -ArgumentList "/install", "/passive", "/norestart" -Wait -PassThru
    if ($process.ExitCode -notin @(0, 1638, 3010)) {
        throw "VC++ runtime installer failed with exit code $($process.ExitCode)."
    }

    $updatedVersion = $null
    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        $updatedVersion = Get-VcRuntimeVersion
        if ($updatedVersion -and $updatedVersion -ge $MinimumVersion) {
            break
        }
        Start-Sleep -Seconds 2
    }

    if (-not $updatedVersion -or $updatedVersion -lt $MinimumVersion) {
        throw "VC++ runtime setup did not reach the required version. Current version: $updatedVersion"
    }

    Write-Step "VC++ x64 runtime is ready ($updatedVersion)."
}

function Invoke-Step {
    param(
        [string]$PythonExe,
        [string[]]$Arguments
    )

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $PythonExe $($Arguments -join ' ')"
    }
}

function Test-PackageVersion {
    param(
        [string]$PythonExe,
        [string]$PackageName,
        [string]$ExpectedVersion
    )

    & $PythonExe -c @"
import importlib.metadata as metadata
import sys

try:
    version = metadata.version('$PackageName')
except metadata.PackageNotFoundError:
    raise SystemExit(1)

raise SystemExit(0 if version == '$ExpectedVersion' else 2)
"@ 1>$null 2>$null

    return ($LASTEXITCODE -eq 0)
}

$repoRoot = Get-RepoRoot
$venvRoot = Get-AbsolutePath -BasePath $repoRoot -CandidatePath $VenvPath
$requirementsPath = Join-Path $repoRoot "configs\deepgaze-runtime-requirements.txt"
$deepgazeArchiveUrl = "https://github.com/matthias-k/DeepGaze/archive/c87b106e8698497c59998b469c45770e993baca3.zip"
$deepgazeRequirement = "deepgaze_pytorch @ $deepgazeArchiveUrl"

Push-Location $repoRoot
try {
    if (-not $SkipVcRuntime) {
        Ensure-VcRuntime
    } else {
        $detectedVersion = Get-VcRuntimeVersion
        Write-Step "Skipping VC++ runtime update. Detected version: $detectedVersion"
    }

    $python312 = Find-Python312
    Write-Step "Using Python 3.12 from $python312"

    if ($ForceRecreate -and (Test-Path $venvRoot)) {
        Write-Step "Removing existing runtime venv at $venvRoot"
        Remove-Item -Recurse -Force $venvRoot
    }

    if (-not (Test-Path $venvRoot)) {
        Write-Step "Creating runtime venv at $venvRoot"
        & $python312 -m venv $venvRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create venv at $venvRoot"
        }
    } else {
        Write-Step "Reusing existing runtime venv at $venvRoot"
    }

    $venvPython = Join-Path $venvRoot "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Expected runtime Python at $venvPython"
    }

    Write-Step "Upgrading pip tooling"
    Invoke-Step -PythonExe $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<81")

    Write-Step "Installing gaze-toolkit base package into the runtime venv"
    Invoke-Step -PythonExe $venvPython -Arguments @("-m", "pip", "install", "-e", ".")

    Write-Step "Installing pinned DeepGaze runtime dependencies"
    Invoke-Step -PythonExe $venvPython -Arguments @("-m", "pip", "install", "-r", $requirementsPath)

    Write-Step "Installing pinned DeepGaze package from GitHub archive"
    Invoke-Step -PythonExe $venvPython -Arguments @("-m", "pip", "install", "--no-deps", $deepgazeRequirement)

    if (Test-PackageVersion -PythonExe $venvPython -PackageName "clip" -ExpectedVersion "1.0") {
        Write-Step "CLIP 1.0 is already installed."
    } else {
        Write-Step "Installing OpenAI CLIP from the validated source archive"
        Invoke-Step -PythonExe $venvPython -Arguments @(
            "-m",
            "pip",
            "install",
            "clip @ https://github.com/openai/CLIP/archive/ded190a052fdf4585bd685cee5bc96e0310d2c93.zip"
        )
    }

    if (-not $SkipSmokeTest) {
        Write-Step "Running DeepGaze runtime smoke test"
        Invoke-Step -PythonExe $venvPython -Arguments @(
            (Join-Path $repoRoot "src\deepgaze_worker.py"),
            "--self-check"
        )
    }

    if ($RunFullValidation) {
        Write-Step "Running full DeepGaze inference validation"
        Invoke-Step -PythonExe $venvPython -Arguments @(
            (Join-Path $repoRoot "scripts\deepgaze_full_validation.py")
        )
    }

    Write-Step "DeepGaze runtime is ready."
    Write-Host "Runtime Python: $venvPython"
    Write-Host "Quick check: $venvPython -c `"from gaze_toolkit.saliency import probe_deepgaze_runtime; print(probe_deepgaze_runtime())`""
} finally {
    Pop-Location
}
