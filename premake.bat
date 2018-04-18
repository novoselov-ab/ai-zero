premake5 vs2017
@if %ERRORLEVEL% neq 0 goto ERROR

:: Success
@exit /B 0

:ERROR
@echo !!! Failure while building !!!
@exit /B 1