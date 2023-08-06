# treetool


The main objective of our work is to estimate the carbon content of trees in a forest plot. With this in mind, we have implemented the algorithm of Liang et al [1] to detect trees in the wild and measure their diameters at breast height (1.3 m) from point clouds. We usually get our point clouds from photogrammetry techniques, where cameras or direct 3D measurements are used, either with LiDAR or TLS sensors. This document describes the requirements, installation, and how to run our code to detect trees and measure their diameter at chest height.

# Software description

treetool is made up of some sample notebooks and three libraries segTree, treetool and Utils. segTree contains several useful functions that allow us to quickly perform operations on point clouds. treetool contains our main class. When called, it performs the complete process of tree detection and extraction of diameters at chest height. Finally, Utils contains special functions required by the internal processes and functions for displaying point clouds.

# Installation

# Hardware requirements
The requirements will depend mainly on the size of the point clouds that will be worked with, since a cloud can vary between hundreds of thousands to hundreds of millions of points. The recommended requirements are designed to process clouds with around ten million points with good fluency.
Minimum technical requirements
• Windows 10
Processor: 1 GHz or more
• RAM: 8 GB
• Graphics: DirectX9 or newer

Recommended technical requirements
• Windows 10
• Processor: 2 GHz or more, 2 or more cores
• RAM: 16 GB
• Graphics: Dedicated video card with 4GB of VRAM

# Software requirements
Anaconda
Python 3.8

# Install Anaconda
Anaconda is a package manager system that may make you life easier with your setup. If you do not have it installed in your system, you may want to download it and install it from https://www.anaconda.com/products/individual/get-started


# Create a Virtual Environment

This may be useful to isolate the package installation to your current system setup.

Create a virtual environment using our environment.yml

```
conda env create
```

To activate the virtual environment run
```
conda activate treetool
```


Finally you can download the point clouds for TLS tests at this address.
[https://drive.google.com/drive/folders/15aW3Npr9lOdxGrswWrsN9wN0g2Q9pBGo?usp=sharing](https://drive.google.com/drive/folders/1AhYd8pwCrTAJCV4OIc9MWot8tjNgIJSx?usp=sharing)

The original databases and original publication can be found on this page.
https://laserscanning.fi/results-available-for-international-benchmarking-of-terrestrial-laser-scanning-methods/


[1] Liang, X., Litkey, P., Hyyppa, J., Kaartinen, H., Vastaranta, M., & Holopainen, M. (2011). Automatic stem mapping using single-scan terrestrial laser scanning. IEEE Transactions on Geoscience and Remote Sensing, 50 (2), 661-670.
