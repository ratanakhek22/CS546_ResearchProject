@echo off
setlocal

set DATASET=%1
set TRIALS=%2
set EF=32
for /f %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'"') do set RUN_ID=%%i

if "%DATASET%"=="" (
    echo Usage: run_experiments.bat [dataset] [trials]
    echo Example: run_experiments.bat scifact 5
    exit /b 1
)

if "%TRIALS%"=="" (
    echo Usage: run_experiments.bat [dataset] [trials]
    echo Example: run_experiments.bat scifact 5
    exit /b 1
)

echo ================================
echo Dataset: %DATASET%
echo Trials:  %TRIALS%
echo EF:      %EF%
echo ================================

echo.
echo [1/7] Running random insertion...
python run_random.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: run_random.py & exit /b 1 )

echo.
echo [2/7] Running k-means insertion...
python run_kmeans.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: run_kmeans.py & exit /b 1 )

echo.
echo [3/7] Running Hilbert curve insertion...
python run_hilbertcurve.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: run_hilbert.py & exit /b 1 )

echo.
echo [4/7] Evaluating results...
python run_lids.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: evaluate.py & exit /b 1 )

echo.
echo [5/7] Evaluating results...
python run_approx_density.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: evaluate.py & exit /b 1 )

echo.
echo [6/7] Evaluating results...
python evaluate.py --dataset %DATASET% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: evaluate.py & exit /b 1 )

echo.
echo [7/7] Generating report...
python ..\data_visualizer\visualize.py --dataset %DATASET% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: visualize.py & exit /b 1 )

echo.
echo ================================
echo Done. Report saved to reports\%DATASET%_*.html
echo ================================

for /f %%i in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'"') do set RUN_ID=%%i
set EF=200

echo ================================
echo Dataset: %DATASET%
echo Trials:  %TRIALS%
echo EF:      %EF%
echo ================================

echo.
echo [1/7] Running random insertion...
python run_random.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: run_random.py & exit /b 1 )

echo.
echo [2/7] Running k-means insertion...
python run_kmeans.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: run_kmeans.py & exit /b 1 )

echo.
echo [3/7] Running Hilbert curve insertion...
python run_hilbertcurve.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: run_hilbert.py & exit /b 1 )

echo.
echo [4/7] Evaluating results...
python run_lids.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: evaluate.py & exit /b 1 )

echo.
echo [5/7] Evaluating results...
python run_approx_density.py --dataset %DATASET% --efconstruction %EF% --trials %TRIALS% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: evaluate.py & exit /b 1 )

echo.
echo [6/7] Evaluating results...
python evaluate.py --dataset %DATASET% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: evaluate.py & exit /b 1 )

echo.
echo [7/7] Generating report...
python ..\data_visualizer\visualize.py --dataset %DATASET% --run-id %RUN_ID%
if %errorlevel% neq 0 ( echo FAILED: visualize.py & exit /b 1 )

echo.
echo ================================
echo Done. Report saved to reports\%DATASET%_*.html
echo ================================