@echo off
setlocal
cd /d %~dp0
set "ROCM=C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
conformer_optimized_hip.exe data out_optimized_hip benchmark 7 5 > optimized_run.log 2>&1
exit /b %ERRORLEVEL%
