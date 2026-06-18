import subprocess
import sys

def run_check():
    result = subprocess.run([sys.executable, "-c", "import pptx; print('installed')"], capture_output=True, text=True)
    if result.returncode == 0:
        print("python-pptx is installed")
    else:
        print("python-pptx is not installed")

if __name__ == "__main__":
    run_check()
