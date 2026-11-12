param(
    [switch]$SkipMNN,
    [string]$OutputRoot,
    [string]$PythonPath,
    [string]$ModelBundle
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Windows x64 is required.' }
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = [IO.Path]::GetFullPath((Join-Path $root '../..'))
$runs = [IO.Path]::GetFullPath((Join-Path $repo 'runs'))
$version = 'v1.3.0-presentation-experimental'
$exeName = 'HeT-OvenCapture-' + $version
if (!$OutputRoot) { $OutputRoot = Join-Path $runs ('desk-build-' + [guid]::NewGuid().ToString('N')) }
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (!$OutputRoot.StartsWith($runs + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputRoot must be a NEW directory inside this repository runs/.'
}
if (Test-Path -LiteralPath $OutputRoot) { throw 'Output exists; preserve it and choose a new OutputRoot.' }
if (!$ModelBundle) { $ModelBundle = Join-Path $runs 'oven-r2-demo-20261112/delivery' }
$ModelBundle = [IO.Path]::GetFullPath($ModelBundle)
$catalog = Join-Path $ModelBundle 'experimental_models.json'
if (!(Test-Path -LiteralPath $catalog)) { throw 'Validated v2/R1/R2 ModelBundle is required.' }
Push-Location -LiteralPath $root
try {
if ($PythonPath) {
    $python = [IO.Path]::GetFullPath($PythonPath)
    if (!(Test-Path -LiteralPath $python)) { throw 'PythonPath does not exist.' }
} else {
    $venv = Join-Path $root '.build-venv'
    $python = Join-Path $venv 'Scripts/python.exe'
    if (!(Test-Path -LiteralPath $python)) {
        & py -3.12 -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw 'Isolated Python 3.12 environment creation failed.' }
    }
}
function Run-Python {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Command failed: ' + ($Arguments -join ' ')) }
}
if (!$PythonPath) {
    $deps = 'requirements-windows.txt'
    if ($SkipMNN) { $deps = 'requirements-windows-no-mnn.txt' }
    Run-Python -Arguments @('-m','pip','install','--disable-pip-version-check','-r',$deps)
}
Run-Python -Arguments @('-m','unittest','discover','-s','tests','-v')
$verify = 'import sys; from pathlib import Path; from model_catalog import read_catalog,resolve_profile; c=read_catalog(Path(sys.argv[1])); assert {p["id"] for p in c.values()}=={"v2","R1","R2"}; [resolve_profile(p) for p in c.values()]; print("Catalog hashes: PASS")'
Run-Python -Arguments @('-c',$verify,$catalog)
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$dist = Join-Path $OutputRoot 'dist'
$work = Join-Path $OutputRoot 'work'
$spec = Join-Path $OutputRoot 'spec'
$buildArgs = @(
    '-m','PyInstaller','--noconfirm','--clean','--onefile','--windowed',
    '--name',$exeName,'--distpath',$dist,'--workpath',$work,'--specpath',$spec,
    '--paths',$root,'--hidden-import','serial.tools.list_ports_windows',
    '--hidden-import','onnxruntime','--collect-binaries','onnxruntime',
    '--collect-all','tkinterdnd2','--collect-all','cv2_enumerate_cameras',
    '--exclude-module','torch','--exclude-module','ultralytics','--exclude-module','MNN.tools'
)
if ($SkipMNN) { $buildArgs += @('--exclude-module','MNN') }
else { $buildArgs += @('--hidden-import','MNN','--collect-binaries','MNN','--collect-data','MNN') }
$buildArgs += @('oven_capture_gui.py')
Run-Python -Arguments $buildArgs
$exe = Join-Path $dist ($exeName + '.exe')
if (!(Test-Path -LiteralPath $exe) -or (Get-Item -LiteralPath $exe).Length -lt 1MB) { throw 'EXE missing or empty.' }
$delivery = Join-Path $OutputRoot 'delivery'
New-Item -ItemType Directory -Path $delivery | Out-Null
Copy-Item -LiteralPath $exe -Destination $delivery
Copy-Item -LiteralPath $catalog -Destination $delivery
foreach ($id in @('v2','r1','r2')) {
    $target = Join-Path $delivery ('models/' + $id)
    New-Item -ItemType Directory -Path $target | Out-Null
    foreach ($name in @('oven12.onnx','classes.json')) {
        Copy-Item -LiteralPath (Join-Path $ModelBundle ('models/' + $id + '/' + $name)) -Destination $target
    }
}
Copy-Item -LiteralPath (Join-Path $root '展示版使用说明.txt') -Destination $delivery
$health = Join-Path $OutputRoot 'healthcheck.json'
$process = Start-Process -FilePath $exe -ArgumentList ('--healthcheck "' + $health + '"') -WindowStyle Hidden -Wait -PassThru
if ($process.ExitCode -ne 0 -or !(Test-Path -LiteralPath $health)) { throw 'Packaged healthcheck failed.' }
$j = Get-Content -LiteralPath $health -Raw | ConvertFrom-Json
if ($j.status -ne 'PASS' -or $j.checks.onnxruntime -like 'UNAVAILABLE:*' -or $j.checks.camera_enumeration -ne 'IMPORT_OK') { throw 'Packaged dependency check failed.' }
if (!$SkipMNN -and $j.checks.MNN -ne 'IMPORT_OK') { throw 'Packaged MNN import failed.' }
Copy-Item -LiteralPath $health -Destination (Join-Path $delivery '构建自检.json')
$lines = Get-ChildItem -LiteralPath $delivery -Recurse -File | Sort-Object FullName | ForEach-Object {
    (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $_.FullName.Substring($delivery.Length + 1).Replace('\','/')
}
Set-Content -LiteralPath (Join-Path $delivery 'SHA256SUMS.txt') -Value $lines -Encoding utf8
$zip = Join-Path $OutputRoot ($exeName + '.zip')
Compress-Archive -Path (Join-Path $delivery '*') -DestinationPath $zip
Write-Host ('EXE: ' + (Join-Path $delivery ($exeName + '.exe')))
Write-Host ('ZIP: ' + $zip)
Write-Host 'Build and dependency self-check passed. Native clicks and hardware are separate acceptance steps.'
} finally { Pop-Location }
