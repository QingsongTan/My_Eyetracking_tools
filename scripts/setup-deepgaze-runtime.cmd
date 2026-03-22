@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-deepgaze-runtime.ps1" %*
exit /b %ERRORLEVEL%
