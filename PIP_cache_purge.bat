@echo off

REM Check if the venv3.10\Scripts\activate.bat file exists
if exist ".\venv3.10\Scripts\activate.bat" (
    REM Call the activate.bat file
    call ".\venv3.10\Scripts\activate.bat"
) else (
    echo venv3.10\Scripts\activate.bat file not found.
    goto :eof
)

REM Purge the pip cache of all tensorrt packages
pip cache remove tensorrt*.*