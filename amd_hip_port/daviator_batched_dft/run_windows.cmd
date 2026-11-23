@echo off
setlocal
cd /d %~dp0
set "ROCM=C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
davidson.exe data out_seq seq 100 > seq.log 2>&1
if errorlevel 1 exit /b 1
davidson.exe data out_batch batch 100 > batch.log 2>&1
exit /b %ERRORLEVEL%
