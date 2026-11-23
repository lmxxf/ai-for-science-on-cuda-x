@echo off
setlocal
cd /d %~dp0
set "ROCM=C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
resident.exe data64 resident_64_seq seq 100 >> resident_run.log 2>&1
if errorlevel 1 exit /b 1
resident.exe data64 resident_64_batch batch 100 >> resident_run.log 2>&1
if errorlevel 1 exit /b 1
resident.exe data256 resident_256_seq seq 100 >> resident_run.log 2>&1
if errorlevel 1 exit /b 1
resident.exe data256 resident_256_batch batch 100 >> resident_run.log 2>&1
if errorlevel 1 exit /b 1
exit /b %ERRORLEVEL%
