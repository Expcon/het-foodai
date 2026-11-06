@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".build-venv\Scripts\python.exe" (
    ".build-venv\Scripts\python.exe" "oven_capture_gui.py"
) else (
    echo No .build-venv detected. Install requirements in your own Python environment, or run the EXE build first.
    py -3.12 "oven_capture_gui.py"
)
if errorlevel 1 pause
