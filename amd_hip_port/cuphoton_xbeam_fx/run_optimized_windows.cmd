@echo off
setlocal
cd /d %~dp0
set "ROCM=C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
fx_optimized.exe data optimized_stable_hip 7 5 > run_optimized.log 2>&1
exit /b %ERRORLEVEL%
