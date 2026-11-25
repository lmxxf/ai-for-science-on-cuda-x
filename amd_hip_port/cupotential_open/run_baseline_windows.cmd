@echo off
setlocal
cd /d %~dp0
set "ROCM=C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"
open_nonbonded_optimized.exe 32000 optimized_baseline_32000 opt 7 5 > run_baseline.log 2>&1
if errorlevel 1 exit /b 1
open_nonbonded_optimized.exe 128000 optimized_baseline_128000 opt 7 5 >> run_baseline.log 2>&1
exit /b %ERRORLEVEL%
