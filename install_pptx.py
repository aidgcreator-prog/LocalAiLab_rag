import subprocess
import sys

def check_pptx():
    try:
        import pptx
        return True
    except ImportError:
        return False

if __name__ == "__main__":
    if check_pptx():
        print("python-pptx is installed")
    else:
        print("python-pptx is NOT installed")
        # Try to install it
        print("Attempting to install python-pptx...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])
            print("python-pptx installed successfully")
        except Exception as e:
            print(f"Failed to install python-pptx: {e}")
