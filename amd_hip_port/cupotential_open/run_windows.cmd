@echo off
setlocal
cd /d %~dp0
set "ROCM=C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
open_nonbonded_hip.exe 32000 out_32000 7 > run.log 2>&1
if errorlevel 1 exit /b 1
open_nonbonded_hip.exe 128000 out_128000 7 >> run.log 2>&1
exit /b %ERRORLEVEL%
