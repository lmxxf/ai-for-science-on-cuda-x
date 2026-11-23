@echo off
setlocal
cd /d %~dp0
set "ROCM=C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
cuspa_hip.exe ..\data ..\out_amd 7 > run.log 2>&1
exit /b %ERRORLEVEL%
