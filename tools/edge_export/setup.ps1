$ErrorActionPreference = 'Stop'
$edgeRepo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$edgeEnv = Join-Path $edgeRepo 'runs\phase2-edge\.venv'
$edgePython = Join-Path $edgeEnv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $edgePython)) {
    & py -3.12 -m venv $edgeEnv
    if ($LASTEXITCODE -ne 0) { throw 'isolated environment creation failed' }
}
& $edgePython -c "import sys; from pathlib import Path; assert sys.prefix != sys.base_prefix; assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()" $edgeEnv
if ($LASTEXITCODE -ne 0) { throw 'unexpected environment; refusing package installation' }
& $edgePython -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r (Join-Path $PSScriptRoot 'requirements-cpu.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'isolated package installation failed' }
Write-Output 'Environment ready. Run python -m tools.edge_export.smoke using the absolute isolated interpreter.'
