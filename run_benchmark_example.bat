@echo off
REM Quick start example for running Rosetta Stone benchmarks (Windows)

echo =======================================================================
echo   ROSETTA STONE BENCHMARK - QUICK START EXAMPLE
echo =======================================================================
echo.

REM Configuration
set FUSION_DB=fusion_db
set SOURCE_DATABASE=examples\database
set TEST_QUERIES=test_queries_benchmark
set NUM_QUERIES=10
set DEVICE=cuda

REM Step 1: Check if fusion database exists
echo [1/3] Checking fusion database...
if not exist "%FUSION_DB%" (
    echo   X Fusion database not found at: %FUSION_DB%
    echo   -^> Please build the fusion database first:
    echo     python merizo_search\merizo.py rosetta build %SOURCE_DATABASE% %FUSION_DB% -d %DEVICE%
    exit /b 1
) else (
    echo   √ Fusion database found
)

REM Step 2: Prepare test queries
echo.
echo [2/3] Preparing test queries...
if not exist "%TEST_QUERIES%" (
    echo   Creating test query set (randomly sampling %NUM_QUERIES% proteins^)...
    python prepare_test_queries.py ^
        --source "%SOURCE_DATABASE%" ^
        --output "%TEST_QUERIES%" ^
        --num %NUM_QUERIES%

    if errorlevel 1 (
        echo   X Failed to prepare test queries
        exit /b 1
    )
) else (
    echo   √ Test queries already exist at: %TEST_QUERIES%
    dir /b "%TEST_QUERIES%\*.pdb" 2>nul | find /c ".pdb" > temp_count.txt
    set /p count=<temp_count.txt
    del temp_count.txt
    echo     Found PDB files in directory
)

REM Step 3: Run speed benchmark
echo.
echo [3/3] Running speed benchmark...
set /a minutes=%NUM_QUERIES% * 30 / 60
echo   This will take approximately %minutes% minutes...
echo.

python benchmark_rosetta.py ^
    --mode speed ^
    --fusion-db "%FUSION_DB%" ^
    --test-queries "%TEST_QUERIES%" ^
    --device "%DEVICE%" ^
    --output-dir benchmark_results ^
    --verbose

if errorlevel 1 (
    echo.
    echo X Benchmark failed. Check benchmark.log for details.
    exit /b 1
)

echo.
echo =======================================================================
echo   BENCHMARK COMPLETE!
echo =======================================================================
echo.
echo Results saved to: benchmark_results\
echo.
echo View results:
echo   - Report:     type benchmark_results\speed_benchmark_report.md
echo   - JSON:       type benchmark_results\speed_benchmark_results.json
echo   - Per-query:  type benchmark_results\speed_benchmark_per_query.csv
echo.

pause
