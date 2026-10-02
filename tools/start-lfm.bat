@echo off
rem Start the CHAT-µ LFM server (port 17889) without the full KHANARY runtime.
rem Used by the distillation factory and standalone LFM testing.
rem
rem Usage:
rem   tools\start-lfm.bat           # start CHAT-µ (LFM2.5-1.2B, port 17889)
rem   tools\start-lfm.bat --tool    # start TOOL-µ (gemma-3-1B, port 17890) instead

setlocal EnableExtensions EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
for %%R in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fR"

if not defined KHANARY_CHAT_MU_MODEL set "KHANARY_CHAT_MU_MODEL=C:\Users\canna\.lmstudio\models\lmstudio-community\LFM2.5-1.2B-Instruct-GGUF\LFM2.5-1.2B-Instruct-Q8_0.gguf"
if not defined KHANARY_CHAT_MU_PORT  set "KHANARY_CHAT_MU_PORT=17889"
if not defined KHANARY_TOOL_MU_MODEL set "KHANARY_TOOL_MU_MODEL=C:\Users\canna\.lmstudio\models\lmstudio-community\gemma-3-1B-it-QAT-GGUF\gemma-3-1B-it-QAT-Q4_0.gguf"
if not defined KHANARY_TOOL_MU_PORT  set "KHANARY_TOOL_MU_PORT=17890"

if not defined KHANARY_LLAMA_SERVER_EXE (
    set "KHANARY_LLAMA_SERVER_EXE=%REPO_ROOT%\build\llama-haswell\bin\llama-server.exe"
)
if not exist "%KHANARY_LLAMA_SERVER_EXE%" (
    set "KHANARY_LLAMA_SERVER_EXE=E:\models\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama-server.exe"
)

set "MU_MODEL=%KHANARY_CHAT_MU_MODEL%"
set "MU_PORT=%KHANARY_CHAT_MU_PORT%"
set "MU_LABEL=chat-mu"
set "MU_KEY=chat_mu"

if /I "%~1"=="--tool" (
    set "MU_MODEL=%KHANARY_TOOL_MU_MODEL%"
    set "MU_PORT=%KHANARY_TOOL_MU_PORT%"
    set "MU_LABEL=tool-mu"
    set "MU_KEY=tool_mu"
)

where pwsh.exe >nul 2>nul
if errorlevel 1 (
    echo [error] pwsh.exe not found
    exit /b 1
)

if not exist "%MU_MODEL%" (
    echo [error] Model not found: %MU_MODEL%
    exit /b 1
)

rem Check if already our server
pwsh.exe -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:%MU_PORT%/v1/models' -Method Get -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 (
    echo [%MU_LABEL%] Already running at http://127.0.0.1:%MU_PORT%
    exit /b 0
)

rem Check if port is taken by something else — find a free one if so
pwsh.exe -NoProfile -Command "try { $l=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Any,%MU_PORT%); $l.Start(); $l.Stop(); exit 0 } catch { exit 1 }" >nul 2>nul
if errorlevel 1 (
    echo [%MU_LABEL%] Port %MU_PORT% is taken — scanning for free port...
    pwsh.exe -NoProfile -Command ^
      "$p=%MU_PORT%; $max=$p+19; while($p -le $max){ try{ $l=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Any,$p); $l.Start(); $l.Stop(); Write-Output $p; exit 0 }catch{ $p++ } }; Write-Output %MU_PORT%" ^
      > "%TEMP%\khanary_lfm_port.tmp" 2>nul
    set /p MU_PORT=<"%TEMP%\khanary_lfm_port.tmp"
    del "%TEMP%\khanary_lfm_port.tmp" >nul 2>nul
    echo [%MU_LABEL%] Using port %MU_PORT%
)

if not exist "%KHANARY_LLAMA_SERVER_EXE%" (
    echo [error] llama-server.exe not found.
    echo         Build it: tools\build_mu_server.bat
    echo         Or set KHANARY_LLAMA_SERVER_EXE to its path.
    exit /b 1
)

for %%D in ("%KHANARY_LLAMA_SERVER_EXE%") do set "MU_EXE_DIR=%%~dpD"
echo [%MU_LABEL%] Starting %MU_LABEL% (port %MU_PORT%)...
echo              Model: %MU_MODEL%
echo              Server: %KHANARY_LLAMA_SERVER_EXE%

rem Write resolved port into data/ports.json (merge — preserve other keys)
set "PORTS_FILE=%~dp0..\data\ports.json"
pwsh.exe -NoProfile -Command ^
  "$f='%PORTS_FILE%'; $j=if(Test-Path $f){try{Get-Content $f -Raw|ConvertFrom-Json}catch{[pscustomobject]@{}}}else{[pscustomobject]@{}}; $j|Add-Member -Force -Name '%MU_KEY%' -Value %MU_PORT% -MemberType NoteProperty; $j|ConvertTo-Json -Compress|Set-Content $f" 2>nul

pwsh.exe -NoProfile -Command "Start-Process -FilePath '%KHANARY_LLAMA_SERVER_EXE%' -ArgumentList @('--model','%MU_MODEL%','--port','%MU_PORT%','--ctx-size','2048','--log-disable') -WorkingDirectory '%MU_EXE_DIR%' -WindowStyle Hidden"
if errorlevel 1 (
    echo [error] Failed to launch server
    exit /b 1
)

echo [%MU_LABEL%] Waiting for server to be ready...
for /l %%I in (1,1,20) do (
    pwsh.exe -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:%MU_PORT%/v1/models' -Method Get -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
    if not errorlevel 1 (
        echo [%MU_LABEL%] Ready at http://127.0.0.1:%MU_PORT%
        exit /b 0
    )
    >nul ping 127.0.0.1 -n 2
)
echo [warn] Server still starting — factory will retry connections automatically.
endlocal
exit /b 0
