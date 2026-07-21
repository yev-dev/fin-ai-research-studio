from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent.resolve()
SRC_DIR = ROOT / "src"
README_PATH = ROOT / "README.md"
REQUIREMENTS_PATH = ROOT / "requirements.txt"
REQUIREMENTS_PATH_WIN = ROOT / "requirements-win.txt"


def read_requirements(path: Path) -> list[str]:
    """Parse pinned requirements from pip-compile output."""

    
    requirements: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip pip-compile metadata/flags and inline comments.
        if line.startswith("--"):
            continue
        if line.startswith("-r"):
            continue
        if line.startswith("-e "):
            continue
        if " # " in line:
            line = line.split(" # ", 1)[0].strip()
        requirements.append(line)
    return requirements


setup(
    name="fin_ai",
    version="0.1.0",
    description="AI Financial Analysis tools and dashboard.",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=read_requirements(REQUIREMENTS_PATH),
)
