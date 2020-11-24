# NISTTreeDetector

An important goal for our team is to assess the carbon content in trees. With that  in mind, we have created models to detect trees in the wild and measure their diameters at breast height (1.3 m) from clouds of points. Usually, we obtain our clouds of points from either photogrammetry techniques, where cameras are employed, or direct 3D measurements, either with LiDAR or TLS sensors.  This document describes the requirements, installation and how to run our code to detect trees and measure their diameter at breast height. As input, we have a cloud of points.


# Install Anaconda
Anaconda is a package manager system that may make you life easier with your setup. If you do not have it installed in your system, you may want to download it and install it from https://www.anaconda.com/products/individual/get-started



# Create a Virtual Environment

This may be useful to isolate the package installation to your current system setup.

Create a virtual environment, called venv, under the current directory and use python as your interpreter

```
python -m venv --system-site-packages .\venv
```

To activate the virtual environment run
```
.\venv\Scripts\activate
```

Before installing packages in the virtual environment, upgrade pip with the command that run pip as a library module of python

```
python -m pip install --upgrade pip 
python -m pip list  # show packages installed within the virtual environment

```





# Requirements for Installation

Run the following commands to make sure you have the required pieces of software.

* Firstly, install jupyter notebook, a web-based programming environment for python, and other languages!
```
conda install jupyter
```

* install Open3d, library for 3d point processing
```
pip install open3d
```

* Install conda-forge, a collection of recipes, build infrastructure and distributions for the conda manager.
```
conda config --add channels conda-forge
```
* Install laspy, a library for reading, writing and modifying files in format LAS for LiDAR data.
```
conda install laspy
```
 * Install PDAL (Point Data Abstraction Library), and GDAL (Geospatial Data Abstraction Library) for python. The first one will be useful for dealing with point clouds while the second one handles raster and vector data.
 ```
conda install -c conda-forge pdal python-pdal gdal
```
 * Install pclpy a python wrapper for the pointcloud library (PCL) developed by davidcaron. It makes it easier to install dependencies for pclpy.
```
conda install -c conda-forge -c davidcaron pclpy
```
* Install matplotlib, a python library for plotting static, animated, and interactive visualizations. 
```
conda install matplotlib
```
* Install pandas, a python library for data manipulation and analysis 
```
conda install pandas
```
* Install scipy, a python library for scientific computing 
```
conda install -c anaconda scipy
```

Fill TLS benchmarking Point clouds can be obtained here
https://laserscanning.fi/results-available-for-international-benchmarking-of-terrestrial-laser-scanning-methods/

Link to Downsampled NIST and TLS benchmarking point clouds
https://drive.google.com/drive/folders/15aW3Npr9lOdxGrswWrsN9wN0g2Q9pBGo?usp=sharing

