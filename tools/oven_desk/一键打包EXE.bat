@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==============================================
echo HeT OvenCapture v1.2.0-experimental - Windows x64 EXE Builder
echo ==============================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1"
if errorlevel 1 (
    echo.
    echo BUILD FAILED. See the error above.
    echo If MNN installation fails, rerun build_windows.ps1 with -SkipMNN.
) else (
    echo.
    echo Build finished. Check dist\HeT-OvenCapture-v1.2.0-experimental.exe
)
pause
