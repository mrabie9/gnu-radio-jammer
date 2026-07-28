@echo off
REM Open a gpsk_comms flowgraph in GNU Radio Companion, straight from the source
REM tree -- nothing needs to be installed first.
REM
REM   run_grc.bat                       # anti-jam loopback (no radio needed)
REM   run_grc.bat examples\foo.grc      # a specific flowgraph
REM
REM Run this from a RadioConda Prompt. The Linux equivalent is run_grc.sh.
REM
REM %~dp0 already ends with a backslash, so paths below append directly to it.
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"

set "FLOWGRAPH=%~1"
if "%FLOWGRAPH%"=="" set "FLOWGRAPH=%PROJECT_ROOT%examples\aj_command_loopback.grc"

where python >nul 2>&1
if errorlevel 1 goto :not_activated

where gnuradio-companion >nul 2>&1
if errorlevel 1 goto :not_activated

if not exist "%FLOWGRAPH%" goto :no_flowgraph

REM Both are needed: PYTHONPATH so the generated flowgraph can import
REM gpsk_comms, GRC_BLOCKS_PATH so Companion can find the block definitions.
set "PYTHONPATH=%PROJECT_ROOT%python;%PYTHONPATH%"
set "GRC_BLOCKS_PATH=%PROJECT_ROOT%grc;%GRC_BLOCKS_PATH%"

python -c "from gpsk_comms import aj_command_tx, gmsk_command_tx" >nul 2>&1
if errorlevel 1 goto :import_failed

echo Project:   %PROJECT_ROOT%
echo Flowgraph: %FLOWGRAPH%
if defined GPSK_COMMS_KEY_FILE goto :have_key
if defined GPSK_COMMS_KEY goto :have_key
echo.
echo Note: GPSK_COMMS_KEY_FILE is not set. The loopback example generates its
echo own ephemeral key, but any real link needs a shared key on both ends:
echo     python -m gpsk_comms.security --output link.key
echo     set GPSK_COMMS_KEY_FILE=%%CD%%\link.key

:have_key
echo.
gnuradio-companion "%FLOWGRAPH%"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

:not_activated
echo ERROR: GNU Radio was not found in this command prompt.
echo Open "RadioConda Prompt" from the Start menu, then run this file again.
pause
endlocal
exit /b 1

:no_flowgraph
echo ERROR: Flowgraph not found: %FLOWGRAPH%
pause
endlocal
exit /b 1

:import_failed
echo.
echo ERROR: The gpsk_comms Python blocks could not be imported.
echo Check that %PROJECT_ROOT%python\gpsk_comms exists and numpy is available.
pause
endlocal
exit /b 1
