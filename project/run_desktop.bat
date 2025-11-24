@echo off
echo Dang khoi dong SSH Tunnel...
echo Nhap mat khau khi cua so moi hien len.

start "SSH Tunnel" cmd /k "ssh -L 5432:localhost:5432 -L 18883:localhost:1883 -L 5000:localhost:5000 tuananh@linuxtuananh.zapto.org -p 2222 -N"

echo Cho 8 giay de ket noi tunnel...
timeout /t 8 /nobreak >nul

echo.
echo Dang khoi dong Desktop App...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the auto-reloader (which will run parking_ui.py)
python run.py

pause
