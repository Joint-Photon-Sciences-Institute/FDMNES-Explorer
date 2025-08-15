"""
Setup script for FDMNES Explorer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fdmnes-explorer",
    version="1.0.0",
    author="Joint Photon Sciences Institute",
    author_email="",
    description="Automated parameter space exploration for FDMNES calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Joint-Photon-Sciences-Institute/FDMNES-Explorer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "toml>=0.10.0",
    ],
    entry_points={
        "console_scripts": [
            "fdmnes-explorer=fdmnes_explorer.explorer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["examples/*.toml"],
    },
)