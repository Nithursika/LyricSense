@echo off
echo Starting LyricSense Genre Classifier...
echo ======================================

:: Set environment variables
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-11.0.27.6-hotspot
set HADOOP_HOME=C:\Hadoop
set PATH=%PATH%;%HADOOP_HOME%\bin
set PYTHON_PATH=C:\Python310\python.exe
set PROJECT_DIR=D:\SEM8\BigData\assi\LyricSense

:: Create project directories if they don't exist
if not exist "%PROJECT_DIR%\templates" mkdir "%PROJECT_DIR%\templates"

:: Change to project directory
cd /d %PROJECT_DIR%

:: Check if the model exists, if not show a helpful message
if not exist "%PROJECT_DIR%\trained_model" (
    echo Model not found! Please train the model first using train_model.py
    echo.
    set /p train_choice="Do you want to train the model now? (y/n): "
    if /i "%train_choice%"=="y" (
        echo Training model...
        %PYTHON_PATH% train_model.py
    ) else (
        echo Exiting...
        pause
        exit /b 1
    )
)

echo Starting Flask application...
echo.
echo Once the server is running, a browser window will open automatically.
echo Press Ctrl+C in this window to stop the application when you're done.
echo.

:: Start the Flask app
%PYTHON_PATH% app.py > flask_output.log 2>&1 &

:: Wait a moment for the Flask server to start
echo Waiting for server to start...
timeout /t 4 /nobreak > nul

:: Use PowerShell to check if the server is up and running
:CheckServer
powershell -Command "try { $null = (New-Object System.Net.Sockets.TcpClient('localhost', 5000)).Close(); exit 1 } catch { exit 0 }"
if %ERRORLEVEL% EQU 1 (
    goto OpenBrowser
) else (
    echo Waiting for Flask server...
    timeout /t 1 /nobreak > nul
    goto CheckServer
)

:OpenBrowser
:: Open the browser to the application URL
echo Opening application in browser...
start http://localhost:5000

:: Keep the window open
echo.
echo LyricSense is now running!
echo The Flask server is running in the background.
echo Press Ctrl+C to stop the server and exit.

:: Use the type command to display log output in real-time
type flask_output.log