@echo off
echo ========================================
echo Running English to Bari MT Project
echo ========================================
echo.

:: Set Python path
set PYTHONPATH=%PYTHONPATH%;C:\Users\Lokudu James\english-bari-mt

:: Activate environment
call conda activate english_bari_mt

:: Step 2: Preprocess data
echo.
echo [Step 2] Preprocessing data...
python scripts\02_preprocess_data.py
if %errorlevel% neq 0 (
    echo Error in preprocessing!
    pause
    exit /b %errorlevel%
)

:: Step 3: Train tokenizer
echo.
echo [Step 3] Training tokenizer...
python scripts\03_train_tokenizer.py
if %errorlevel% neq 0 (
    echo Error in tokenizer training!
    pause
    exit /b %errorlevel%
)

:: Step 4: Train model
echo.
echo [Step 4] Training model (this will take 2-3 hours)...
python scripts\04_train_model.py
if %errorlevel% neq 0 (
    echo Error in model training!
    pause
    exit /b %errorlevel%
)

:: Step 5: Evaluate model
echo.
echo [Step 5] Evaluating model...
python scripts\05_evaluate_model.py
if %errorlevel% neq 0 (
    echo Error in evaluation!
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo Project completed successfully!
echo ========================================
pause