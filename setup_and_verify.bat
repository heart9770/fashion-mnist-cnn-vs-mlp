@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Fashion-MNIST CNN vs MLP - one click setup and verify
echo  (takes about 10-15 minutes, needs internet)
echo ============================================================
echo.

echo [1/6] looking for a supported Python (3.12 / 3.13 / 3.14) ...
set "PYCMD="
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3.12"
if defined PYCMD goto :gotpython
py -3.13 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3.13"
if defined PYCMD goto :gotpython
py -3.14 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3.14"
if defined PYCMD goto :gotpython
goto :nopython

:gotpython
%PYCMD% --version
echo using: %PYCMD%

echo.
echo [2/6] creating virtual environment .venv ...
if not exist ".venv\Scripts\python.exe" (
  %PYCMD% -m venv .venv
  if errorlevel 1 goto :failed
) else (
  echo already exists, reuse it
)
call ".venv\Scripts\activate.bat"

echo.
echo [3/6] installing packages (about 200 MB, please wait) ...
python -m pip install --upgrade pip --quiet
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :failed
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
python -c "import torch; print('torch', torch.__version__)"
if errorlevel 1 goto :failed

echo.
echo [4/6] downloading Fashion-MNIST (about 30 MB) ...
python download_data.py
if errorlevel 1 goto :failed

echo.
echo [5/6] training with seed 42 (about 5 minutes) ...
python train_fashion_mnist.py --out results_check --epochs 12 --seeds 42 --task main
if errorlevel 1 goto :failed

echo.
echo [6/6] comparing with the saved results ...
python verify_repro.py --new results_check --tol 0
echo.
echo ------------------------------------------------------------
echo If every row above says "match", reproduction succeeded.
echo Full guide: REPRODUCE.md????.md
echo ------------------------------------------------------------
pause
exit /b 0

:nopython
echo.
echo No Python 3.12 / 3.13 / 3.14 found on this machine.
echo.
echo PyTorch currently supports Python 3.10 - 3.14 only (NOT 3.15).
echo If you already installed 3.15, install 3.12 as well - both can coexist,
echo the launcher picks the right one by version number.
echo.
if exist "%~dp0python-3.12.10-amd64.exe" (
  echo Opening the installer shipped in this folder ...
  start "" "%~dp0python-3.12.10-amd64.exe"
  echo.
  echo IMPORTANT: tick "Add python.exe to PATH" on the first screen,
  echo then click "Install Now" and finish. After that run this file again.
) else (
  echo Download it from https://www.python.org/downloads/windows/
  echo (choose 3.12.x, NOT the newest 3.15) and tick "Add python.exe to PATH".
)
echo.
pause
exit /b 1

:failed
echo.
echo Something failed. Read the message above, then see section 11
echo of REPRODUCE.md (troubleshooting table).
echo.
pause
exit /b 1
