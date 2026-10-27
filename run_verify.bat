@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Verify the report results (portable Python, no install needed)
echo ============================================================
echo.

if not exist "py312\python.exe" (
  echo [ERROR] py312\python.exe not found. This repo does not ship a Python runtime.
  echo Either install deps with: pip install -r requirements.txt
  echo or build a portable env with: python setup_portable_python.py
  pause
  exit /b 1
)

echo [1/3] checking data ...
if not exist "data\fashion\train-images-idx3-ubyte.gz" (
  echo downloading Fashion-MNIST ...
  "py312\python.exe" download_data.py
  if errorlevel 1 goto :failed
) else (
  echo data already present
)

echo.
echo [2/3] training with seed 42 (about 5 minutes) ...
"py312\python.exe" train_fashion_mnist.py --out results_check --epochs 12 --seeds 42 --task main
if errorlevel 1 goto :failed

echo.
echo [3/3] comparing with the saved results ...
"py312\python.exe" verify_repro.py --new results_check --tol 0
echo.
echo ------------------------------------------------------------
echo Four rows showing "match" mean reproduction succeeded.
echo ------------------------------------------------------------
pause
exit /b 0

:failed
echo.
echo Something failed. See the message above.
pause
exit /b 1
