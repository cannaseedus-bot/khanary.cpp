@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem test.bat [TARGET|path\test.exe] [test arguments...]
rem Stages native test binaries on C: before execution to avoid E: execute denial.

set "SOURCE_DIR=%~dp0"
if "%SOURCE_DIR:~-1%"=="\" set "SOURCE_DIR=%SOURCE_DIR:~0,-1%"
set "TARGET=%~1"
if not defined TARGET set "TARGET=KHANARY_CONTRACT_TEST"

if /i "%TARGET%"=="KHANARY_XCFE_KXML_LOWERING_TEST" (
  echo [test] python "%SOURCE_DIR%\tools\check_xcfe_kxml_lowering.py"
  python "%SOURCE_DIR%\tools\check_xcfe_kxml_lowering.py"
  exit /b !errorlevel!
)

if /i "%TARGET%"=="KHANARY_DAEMON_BOT_TEST" (
  echo [test] python "%SOURCE_DIR%\tools\compile_daemon_bot.py" --check
  pushd "%SOURCE_DIR%"
  python "tools\compile_daemon_bot.py" --check
  set "RC=!errorlevel!"
  popd
  exit /b !RC!
)

if /i "%TARGET:~-4%"==".exe" (
  if exist "%TARGET%" (set "SOURCE_EXE=%~f1") else (set "SOURCE_EXE=%SOURCE_DIR%\bin\%TARGET%")
) else (
  set "SOURCE_EXE=%SOURCE_DIR%\bin\%TARGET%.exe"
)

if not exist "!SOURCE_EXE!" (
  echo [test] Missing binary: !SOURCE_EXE!
  echo [test] Build it first with: build.bat %TARGET%
  exit /b 2
)

set "STAGE=%TEMP%\khanary-test-run\%RANDOM%-%RANDOM%"
mkdir "!STAGE!" >nul 2>&1
copy /y "!SOURCE_EXE!" "!STAGE!\" >nul
for %%D in ("%SOURCE_DIR%\bin\*.dll") do copy /y "%%~fD" "!STAGE!\" >nul
set "STAGED_EXE=!STAGE!\%~nx1"
if "%~1"=="" set "STAGED_EXE=!STAGE!\KHANARY_CONTRACT_TEST.exe"
if not exist "!STAGED_EXE!" for %%E in ("!SOURCE_EXE!") do set "STAGED_EXE=!STAGE!\%%~nxE"

set "ARGS="
if not "%~1"=="" shift
:args
if "%~1"=="" goto defaults
set "ARGS=!ARGS! "%~1""
shift
goto args

:defaults
if not defined ARGS if /i "%TARGET%"=="KHANARY_CONTRACT_TEST" set "ARGS="%SOURCE_DIR%\data\conformance\contracts.json" "%SOURCE_DIR%\data""
if not defined ARGS if /i "%TARGET%"=="KHANARY_ALGEBRA_ISA_TEST" set "ARGS="%SOURCE_DIR%\data""
if not defined ARGS if /i "%TARGET%"=="KHANARY_APP_HOST_RUNTIME_TEST" (
  set "TEST_REPO=!STAGE!\repo"
  set "TEST_WORK=!STAGE!\work"
  mkdir "!TEST_REPO!\data" >nul 2>&1
  mkdir "!TEST_REPO!\drivers" >nul 2>&1
  mkdir "!TEST_WORK!" >nul 2>&1
  xcopy /e /i /q /y "%SOURCE_DIR%\data\host" "!TEST_REPO!\data\host" >nul
  xcopy /e /i /q /y "%SOURCE_DIR%\data\grammar" "!TEST_REPO!\data\grammar" >nul
  xcopy /e /i /q /y "%SOURCE_DIR%\data\programs" "!TEST_REPO!\data\programs" >nul
  mkdir "!TEST_REPO!\data\grammar" >nul 2>&1
  copy /y "%SOURCE_DIR%\data\grammar\css.key-grammar.jsonl" "!TEST_REPO!\data\grammar\" >nul
  copy /y "%SOURCE_DIR%\data\grammar\asx-mini-app.key-grammar.jsonl" "!TEST_REPO!\data\grammar\" >nul
  copy /y "%SOURCE_DIR%\data\grammar\key-grammar-json.runtime.v4.ebnf" "!TEST_REPO!\data\grammar\" >nul
  copy /y "%SOURCE_DIR%\data\grammar\xcfe-kxml-lowering.contracts.json" "!TEST_REPO!\data\grammar\" >nul
  xcopy /e /i /q /y "%SOURCE_DIR%\data\manifest" "!TEST_REPO!\data\manifest" >nul
  mkdir "!TEST_REPO!\data\shaders" >nul 2>&1
  copy /y "%SOURCE_DIR%\data\shaders\svg_compute.hlsl" "!TEST_REPO!\data\shaders\" >nul
  xcopy /e /i /q /y "%SOURCE_DIR%\drivers\micronauts" "!TEST_REPO!\drivers\micronauts" >nul
  mkdir "!TEST_REPO!\drivers\runtime" >nul 2>&1
  copy /y "%SOURCE_DIR%\drivers\runtime\models.toml" "!TEST_REPO!\drivers\runtime\" >nul
  set "ARGS="!TEST_REPO!" "!TEST_WORK!""
)
set "KHANARY_SOURCE_DIR=%SOURCE_DIR%"
pushd "!STAGE!"
echo [test] !STAGED_EXE! !ARGS!
"!STAGED_EXE!" !ARGS!
set "RC=!errorlevel!"
popd
rmdir /s /q "!STAGE!" >nul 2>&1
exit /b !RC!
