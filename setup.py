from setuptools import setup

setup(
    name="treetool",
    version="0.1.0",
    description="A package for tree detection from tls point clouds",
    url="https://github.com/porteratzo/TreeTool",
    author="Omar Montoya",
    author_email="omar.alfonso.montoya@hotmail.com",
    license="MIT",
    packages=["TreeTool"],
    install_requires=[
        open3d,
        lsq - ellipse,
        jupyter,
        matplotlib,
        scipy,
    ],
)
