# Saved with UTF-8 BOM for Windows PowerShell 5.1 compatibility.
param(
    [switch]$SkipMNN
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') {
    throw '必须在 Windows 10/11 x64 上打包 Windows EXE。PyInstaller 无法从 Linux 直接交叉生成 Windows EXE。'
}
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$version = 'v1.2.0-experimental'
$exeName = 'HeT-OvenCapture-' + $version
Set-Location -LiteralPath $root
Write-Host ('Building HeT OvenCapture in: ' + $root) -ForegroundColor Cyan

$venv = Join-Path $root '.build-venv'
$python = Join-Path $venv 'Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) {
    Write-Host 'Creating a dedicated Python 3.12 build environment...' -ForegroundColor Cyan
    $launcher = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        & py -3.12 -m venv $venv
    } else {
        & python -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败。请安装 Python 3.12 x64（勾选 Python Launcher）。' }
}
if (!(Test-Path -LiteralPath $python)) { throw '未找到构建虚拟环境 python.exe' }

function Run-Python {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('命令执行失败: ' + ($Arguments -join ' ')) }
}

Write-Host 'Installing isolated dependencies...' -ForegroundColor Cyan
$deps = 'requirements-windows.txt'
if ($SkipMNN) { $deps = 'requirements-windows-no-mnn.txt' }
Run-Python -Arguments @('-m','pip','install','--disable-pip-version-check','-r',$deps)
Write-Host 'Running core tests without opening a camera...' -ForegroundColor Cyan
Run-Python -Arguments @('-m','unittest','discover','-s','tests','-v')

$dist = Join-Path $root 'dist'
$work = Join-Path $root '.pyinstaller-build'
$spec = Join-Path $root '.pyinstaller-spec'
$pyinstallerArgs = @(
    '-m','PyInstaller','--noconfirm','--clean','--onefile','--windowed',
    '--name',$exeName,'--distpath',$dist,'--workpath',$work,'--specpath',$spec,
    '--paths',$root,
    '--hidden-import','serial.tools.list_ports_windows',
    '--hidden-import','onnxruntime',
    '--collect-binaries','onnxruntime',
    '--collect-all','tkinterdnd2',
    '--collect-all','cv2_enumerate_cameras',
    '--exclude-module','torch',
    '--exclude-module','ultralytics',
    '--exclude-module','MNN.tools'
)
if (-not $SkipMNN) {
    # Do not collect/import MNN.tools: its upstream Python wrapper has previously
    # attempted unwanted package installation. We only need the MNN runtime.
    $pyinstallerArgs += @('--hidden-import','MNN','--collect-binaries','MNN','--collect-data','MNN')
} else {
    $pyinstallerArgs += @('--exclude-module','MNN')
}
$pyinstallerArgs += @('oven_capture_gui.py')
Write-Host 'Building self-contained GUI EXE (first build can take several minutes)...' -ForegroundColor Cyan
Run-Python -Arguments $pyinstallerArgs

$exe = Join-Path $dist ($exeName + '.exe')
if (!(Test-Path -LiteralPath $exe) -or (Get-Item -LiteralPath $exe).Length -lt 1MB) {
    throw '未生成有效的 .exe 文件'
}
$health = Join-Path $root 'build-healthcheck-windows.json'
if (Test-Path -LiteralPath $health) { Remove-Item -LiteralPath $health -Force }
Write-Host 'Starting packaged EXE in safe health-check mode (no camera opened)...' -ForegroundColor Cyan
# A PyInstaller --windowed binary is a GUI-subsystem process on Windows.
# Start-Process -Wait is needed to reliably wait for self-test completion.
$healthArg = '--healthcheck "' + $health + '"'
$healthProcess = Start-Process -FilePath $exe -ArgumentList $healthArg -WindowStyle Hidden -Wait -PassThru
if ($healthProcess.ExitCode -ne 0) { throw 'EXE 运行时自检失败（可能缺少打包的 DLL）。' }
if (!(Test-Path -LiteralPath $health)) { throw 'EXE 没有写出自检 JSON' }
$j = Get-Content -LiteralPath $health -Raw | ConvertFrom-Json
if ($j.status -ne 'PASS') { throw 'EXE 自检 JSON 非 PASS' }
if ($j.checks.onnxruntime -like 'UNAVAILABLE:*') { throw '未打包 ONNX Runtime，不能交付模型功能' }
if ($j.checks.camera_enumeration -ne 'IMPORT_OK') { throw '设备名称枚举组件缺失；请检查 cv2-enumerate-cameras 依赖收集。' }
if ((-not $SkipMNN) -and ($j.checks.MNN -ne 'IMPORT_OK')) {
    throw 'MNN runtime 未能在 EXE 中导入。可以先 -SkipMNN 构建摄像头/ONNX 版。'
}

$release = Join-Path $root ('release-windows-' + $version)
New-Item -ItemType Directory -Path $release -Force | Out-Null
Copy-Item -LiteralPath $exe -Destination (Join-Path $release ($exeName + '.exe')) -Force
Copy-Item -LiteralPath (Join-Path $root '组员使用说明.txt') -Destination (Join-Path $release '组员使用说明.txt') -Force
Copy-Item -LiteralPath $health -Destination (Join-Path $release '构建自检.json') -Force
$hash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
Set-Content -LiteralPath (Join-Path $release 'EXE_SHA256.txt') -Value ($hash + '  ' + $exeName + '.exe') -Encoding ascii
$zip = Join-Path $root ('HeT-OvenCapture-Win64-' + $version + '.zip')
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path (Join-Path $release '*') -DestinationPath $zip
Write-Host ''
Write-Host 'BUILD COMPLETE' -ForegroundColor Green
Write-Host ('EXE: ' + $exe) -ForegroundColor Green
Write-Host ('Shareable ZIP: ' + $zip) -ForegroundColor Green
Write-Host ('SHA256: ' + $hash)
Write-Host 'Note: physical cameras, MNN model semantics and serial ports still require testing on classmates Windows PCs.'
