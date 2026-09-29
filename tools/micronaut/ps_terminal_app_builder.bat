@echo off
setlocal

if "%~1"=="" (
    echo Usage: ps_terminal_app_builder.bat ^<chat^|integration^|ide^|wpf^|svg^|build-app^> [additional args]
    echo Example: ps_terminal_app_builder.bat build-app -TargetRoot ".ui-shell"
    exit /b 1
)

set MODE=%~1
shift

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%ps_terminal_app_builder.ps1"
if not exist "%PS_SCRIPT%" (
    set "PS_SCRIPT=%CD%\tools\micronaut\ps_terminal_app_builder.ps1"
)
if not exist "%PS_SCRIPT%" (
    echo Could not locate ps_terminal_app_builder.ps1
    exit /b 1
)

set "SHELL_EXE=pwsh.exe"
where pwsh.exe >nul 2>nul
if errorlevel 1 set "SHELL_EXE=powershell.exe"

set "FORWARD_ARGS="
:collect_args
if "%~1"=="" goto run_ps
set FORWARD_ARGS=%FORWARD_ARGS% "%~1"
shift
goto collect_args

:run_ps
"%SHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -Mode "%MODE%" %FORWARD_ARGS%
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
