@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"

where python >nul 2>&1
if errorlevel 1 goto :not_activated

where gnuradio-companion >nul 2>&1
if errorlevel 1 goto :not_activated

set "PYTHONPATH=%PROJECT_ROOT%python;%PYTHONPATH%"
set "GRC_BLOCKS_PATH=%PROJECT_ROOT%grc;%GRC_BLOCKS_PATH%"

python -c "from gpsk_comms import gmsk_command_tx; from gpsk_comms.keyboard_command_source import keyboard_command_source; print('gpsk_comms portable blocks: OK')"
if errorlevel 1 goto :import_failed

echo.
echo Opening the portable keyboard transmitter in GNU Radio Companion...
echo Project: %PROJECT_ROOT%
echo.
gnuradio-companion "%PROJECT_ROOT%examples\gmsk_keyboard_tx_uhd.grc"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

:not_activated
echo ERROR: GNU Radio was not found in this command prompt.
echo Open "RadioConda Prompt" from the Start menu, then run this file again.
pause
endlocal
exit /b 1

:import_failed
echo.
echo ERROR: The portable gpsk_comms Python blocks could not be loaded.
echo Make sure the complete extracted folder is present and use RadioConda Prompt.
pause
endlocal
exit /b 1
