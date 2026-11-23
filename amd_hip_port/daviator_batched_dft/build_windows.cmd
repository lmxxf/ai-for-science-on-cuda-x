@echo off
setlocal
cd /d %~dp0
set "ROCM=C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel"
call "D:\DLSSNR-Lab\build-tools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
"%ROCM%\lib\llvm\bin\clang++.exe" -x hip --offload-arch=gfx1201 --rocm-path="%ROCM%" --rocm-device-lib-path="%ROCM%\lib\llvm\amdgcn\bitcode" -std=c++17 -O3 -ffp-contract=off -I"%ROCM%\include" davidson.cpp -o davidson.exe -L"%ROCM%\lib" -lamdhip64 > build.log 2>&1
exit /b %ERRORLEVEL%
