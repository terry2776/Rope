if exist "activate.bat" (
    call activate.bat
)
git checkout -f main
python Rope.py
pause