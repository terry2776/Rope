if exist "activate.bat" (
    call activate.bat
)
git checkout -f development
git reset --hard origin/development
git pull origin development
python .\tools\download_models.py
python .\tools\update_rope.py

echo.
echo --------Update completed--------
echo.

pause