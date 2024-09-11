echo off
cls
echo Welcome to the VepleyAI project!
echo Please wait while we set up your environment...
py -3.9 -m venv VepleyEnv
REM loop until the environment is set up
:loop
if exist VepleyEnv\Scripts\activate.bat goto :activate
timeout 1
goto :loop
:activate
echo Activating VepleyEnv...
echo Environment setup complete!
VepleyEnv\Scripts\python -m pip install -r requirements.txt
cls
echo Starting VepleyAI_acquire.py...
VepleyEnv\Scripts\python VepleyAI_acquire.py