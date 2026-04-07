@echo off
setlocal
cd /d %~dp0

echo === PTX-RADAR Build ===

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
)

if not exist build mkdir build
cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

cmake --build . --config Release
if errorlevel 1 exit /b 1

echo.
echo === Build successful ===
