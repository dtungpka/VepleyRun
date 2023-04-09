REM using pyinstaller to build the executable for VepleyAI_acquire.py
cls
echo off
echo Building VepleyAI_acquire.exe...
REM using pyinstaller with flag: --onefile
pyinstaller VepleyAI_acquire.py

