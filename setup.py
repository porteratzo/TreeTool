import setuptools
from setuptools import setup

setup(
    name="treetool",
    version="1.0.1",
    description="Python package for tree detection, segmentation and extraction of DBH",
    url="https://github.com/porteratzo/TreeTool",
    author="Omar Montoya",
    author_email="omar.alfonso.montoya@hotmail.com",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=[
        "open3d",
        "lsq-ellipse",
        "matplotlib",
        "pandas",
        "numpy>=1.18.0,<2.0.0",
        "scipy",
        "git+https://github.com/porteratzo/porteratzo3D.git",
    ],
    classifiers=[],
)
