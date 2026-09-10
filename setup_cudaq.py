"""Setup script for CMT-QNN on NVIDIA Quantum Stack."""

from setuptools import setup, find_packages

with open("README_CUDAQ.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements_cudaq.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cudaq_cmt",
    version="1.0.0",
    author="CMT-QNN Team",
    description="CMT-QNN experiments on NVIDIA Quantum Stack (CUDA-Q + cuQuantum)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/cmt-code",
    packages=find_packages(include=["cudaq", "cudaq.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "mpi": [
            "mpi4py>=3.1",
        ],
        "docs": [
            "sphinx>=7.0",
            "sphinx-rtd-theme>=1.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "cudaq-cmt=cudaq_main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cudaq": ["*.py"],
    },
)