<#
.SYNOPSIS
    Start the Streamlit app with .env actually taking effect.

.DESCRIPTION
    src/config.py calls load_dotenv(override=False), which keeps process
    environment variables authoritative. A variable left over in the shell
    therefore wins over .env, and editing .env appears to do nothing.

    src/config.py detects and reports this, but detection is not a fix: the
    shadowing variable has to be removed from the process that launches
    Streamlit. This script removes any app setting that is currently shadowing
    .env, prints the configuration that will actually be used, and then starts
    the app. The names come from src.config rather than a hardcoded list, so
    the script cannot drift from what the application reads.

.PARAMETER Port
    Port for the Streamlit server. Defaults to 8501.

.EXAMPLE
    .\start.ps1

.EXAMPLE
    .\start.ps1 -Port 8600
#>
[CmdletBinding()]
param(
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "Virtual environment not found at $python. Run 'uv sync --extra dev' first."
}

if (-not (Test-Path (Join-Path $root '.env'))) {
    Write-Warning 'No .env file found. The app will run on built-in defaults.'
}

# The one stderr message that is expected and must be ignored: src.config.py
# logs a warning per shadowed name, which is precisely the condition this
# script exists to resolve. Matching is done on whitespace-normalised text
# because the message wraps, and it is non-greedy so that any real error
# alongside the warning still surfaces.
$ShadowWarning = 'Environment variable .+? is set in this process .*? to use the file value\.'

function Invoke-Python {
    <#
    .SYNOPSIS
        Run a snippet of Python, returning its stdout as a string array.
    .DESCRIPTION
        PowerShell wraps every line a native command writes to stderr in an
        ErrorRecord, and under $ErrorActionPreference = 'Stop' that aborts the
        script. So the preference is relaxed for the duration of the call, and
        the error handling is done explicitly instead: stderr is filtered
        against the known warning, and anything remaining, or a non-zero exit
        code, is raised as a terminating error.

        Python literals in the snippets passed to this function must use single
        quotes. PowerShell strips double quotes when it builds a native command
        line, so a double-quoted literal reaches Python with its quotes removed
        and the snippet dies with a SyntaxError.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Code
    )

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $captured = & $python -c $Code 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }

    $stdout = @()
    $stderr = @()
    foreach ($item in @($captured)) {
        if ($item -is [System.Management.Automation.ErrorRecord]) {
            $stderr += $item.ToString()
        }
        else {
            $stdout += [string]$item
        }
    }

    # Collapse wrapping and the "python.exe :" / "At ...char:13" decoration that
    # PowerShell adds, so the warning can be recognised as one sentence.
    $noise = ($stderr -join ' ') -replace '\s+', ' '
    $noise = [regex]::Replace($noise, $ShadowWarning, '')
    $noise = ($noise -replace '\s+', ' ').Trim()

    if ($exitCode -ne 0 -or $noise) {
        throw "Python snippet failed (exit $exitCode).`n$($stderr -join [Environment]::NewLine)"
    }

    return $stdout
}

Write-Host 'Checking for environment variables that override .env...' -ForegroundColor DarkGray
$shadowed = @(Invoke-Python -Code "from src.config import shadowed_env_names; print('\n'.join(shadowed_env_names))")

$cleared = @()
foreach ($name in $shadowed) {
    if ([string]::IsNullOrWhiteSpace($name)) { continue }
    Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue
    $cleared += $name
    Write-Host "  cleared Env:$name" -ForegroundColor Yellow
}

if ($cleared.Count -eq 0) {
    Write-Host '  nothing was shadowing .env' -ForegroundColor DarkGray
}
else {
    Write-Host "  .env is now authoritative for $($cleared.Count) setting(s)." -ForegroundColor Green
}

# Read the settings back after clearing, so a model change is confirmed before
# the server starts rather than inferred from behaviour later.
Write-Host ''
Write-Host 'Effective configuration:' -ForegroundColor DarkGray
Invoke-Python -Code @'
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_TOP_K, LLM_PROVIDER, OLLAMA_MODEL
print('  provider   ', LLM_PROVIDER)
if LLM_PROVIDER == 'ollama':
    print('  model      ', OLLAMA_MODEL)
print('  chunking   size=%s overlap=%s' % (CHUNK_SIZE, CHUNK_OVERLAP))
print('  top_k      ', DEFAULT_TOP_K)
'@ | ForEach-Object { Write-Host $_ }

Write-Host ''
Write-Host "Starting Streamlit on http://localhost:$Port" -ForegroundColor Cyan
Write-Host ''
& $python -m streamlit run app.py --server.port $Port
