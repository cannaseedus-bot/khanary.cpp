@echo off
setlocal

set OUTPUT_ROOT=%~1
if "%OUTPUT_ROOT%"=="" set OUTPUT_ROOT=%~dp0..\..\artifacts\micronaut-bundles

set BUNDLE_NAME=%~2
if "%BUNDLE_NAME%"=="" set BUNDLE_NAME=mx2lm-wpf-cli-v1

where pwsh.exe >nul 2>nul
if errorlevel 1 (
    echo PowerShell 7+ ^(pwsh.exe^) is required to build the MX2LM WPF bundle.
    exit /b 1
)

pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_mx2lm_wpf_bundle.ps1" -OutputRoot "%OUTPUT_ROOT%" -BundleName "%BUNDLE_NAME%" -IncludeDatasetRows 1 -Force
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
