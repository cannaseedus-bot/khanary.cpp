@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Build llama-server.exe with Haswell AVX2/FMA/F16C for CHAT-µ / TOOL-µ inference.
rem Output: build\llama-haswell\bin\llama-server.exe

set "SRC=E:\models\.ASX.cpp\llama-b9968-bin-win-cpu-x64\llama.cpp"
set "BUILD=C:\Users\canna\_khanary_inspect\KHANARY.CPP\build\llama-haswell"
set "OUT=%BUILD%\bin"

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "CMAKE=C:\Program Files\CMake\bin\cmake.exe"

if not exist "%SRC%\CMakeLists.txt" (
    echo [error] Source not found: %SRC%
    exit /b 1
)
if not exist "%VCVARS%" (
    echo [error] vcvars64.bat not found: %VCVARS%
    exit /b 1
)
if not exist "%CMAKE%" (
    echo [error] cmake.exe not found: %CMAKE%
    exit /b 1
)

call "%VCVARS%"
if errorlevel 1 ( echo [error] vcvars64 failed & exit /b 1 )

set "NINJA_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "PATH=%PATH%;%NINJA_DIR%"

echo [configure] Haswell AVX2 build...
"%CMAKE%" -S "%SRC%" -B "%BUILD%" ^
    -G "Ninja" ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="%OUT%" ^
    -DGGML_AVX2=ON ^
    -DGGML_NATIVE=OFF ^
    -DGGML_OPENMP=ON ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DLLAMA_BUILD_SERVER=ON ^
    -DLLAMA_BUILD_TESTS=OFF ^
    -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DLLAMA_BUILD_UI=OFF
if errorlevel 1 ( echo [error] cmake configure failed & exit /b 1 )

echo [build] Compiling llama-server (this takes ~5-10 minutes)...
"%CMAKE%" --build "%BUILD%" --config Release --target llama-server -j 4
if errorlevel 1 ( echo [error] build failed & exit /b 1 )

echo.
echo [done] llama-server.exe: %OUT%\llama-server.exe
if exist "%OUT%\llama-server.exe" (
    dir "%OUT%\llama-server.exe"
) else (
    rem Ninja puts output in build root, not subdir
    if exist "%BUILD%\llama-server.exe" (
        echo [done] llama-server.exe: %BUILD%\llama-server.exe
        dir "%BUILD%\llama-server.exe"
    )
)
endlocal
