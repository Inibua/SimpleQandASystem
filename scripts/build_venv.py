import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from paths import ROOT_DIR

VENV_DIR = Path(ROOT_DIR, "venv")
REQUIREMENTS_FILE = Path(Path(__file__).cwd(), "requirements.txt")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def create_venv() -> None:
    run_command([sys.executable, "-m", "venv", str(VENV_DIR)])


def venv_python_path() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_requirements() -> None:
    python_path = venv_python_path()
    if not python_path.exists():
        raise RuntimeError("pip not found in virtual environment")

    run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(python_path), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def clean_venv() -> None:
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or update Python virtual environment")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing venv and recreate it from scratch",
    )
    args = parser.parse_args()

    if args.clean:
        clean_venv()
        create_venv()
    else:
        if not VENV_DIR.exists():
            create_venv()

    install_requirements()


if __name__ == "__main__":
    main()
