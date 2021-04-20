#!/usr/bin/env python3
from setuptools import setup, find_namespace_packages

setup(
    name="vortector",
    version="1.0",
    description="A vortex detector for planet-disk simulation data.",
    author="Thomas Rometsch",
    author_email="thomas.rometsch@uni-tuebingen.de",
    url="https://github.com/rometsch/vortector",
    package_dir={'': 'src'},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "numpy", "matplotlib", "opencv-python"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Utilities"
    ],
    python_requires='>=3.6'
)
