@echo off
setlocal

:: Check for required arguments
if "%~1"=="" (
    echo Usage: create_model.bat ModelFrom ContextLength
    exit /b 1
)

if "%~2"=="" (
    echo Usage: create_model.bat ModelFrom ContextLength
    exit /b 1
)

:: Get arguments
set "MODEL_FROM=%~1"
set "CONTEXT_LENGTH=%~2"

:: Extract model name before colon (if present)
for /f "tokens=1 delims=:" %%a in ("%MODEL_FROM%") do set "MODEL_NAME_BASE=%%a"

:: Build final model name
set "FINAL_MODEL_NAME=%MODEL_NAME_BASE%%CONTEXT_LENGTH%"

:: Create Modelfile
(
    echo FROM %MODEL_FROM%
    echo PARAMETER num_ctx %CONTEXT_LENGTH%
) > Modelfile

echo Modelfile created.

:: Run ollama create
ollama create -f Modelfile %FINAL_MODEL_NAME%

echo Model '%FINAL_MODEL_NAME%' created.
