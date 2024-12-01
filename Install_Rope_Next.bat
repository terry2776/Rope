@echo off
if exist "activate.bat" (
    call activate.bat
)
git init
git remote add origin https://github.com/terry2776/Rope.git
git pull origin
git checkout -f -b main origin/main
git reset --hard origin/main

call PIP_cache_purge.bat
call Update_Rope_Next_Stable.bat

echo.
echo --------Installation Complete--------
echo.
echo You can now start the program by running the Start_Rope_Next_Stable.bat or Start_Rope_Next_Dev.bat file

pause