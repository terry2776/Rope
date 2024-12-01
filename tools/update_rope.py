import subprocess
import sys

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_cu124.txt", "--default-timeout", "100"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "tensorrt-cu12==10.4.0", "--default-timeout", "100"])
install_requirements()
pass
