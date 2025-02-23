#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="dHBV_2_0",
    version="1.0.0",
    description="High-resolution, multiscale, differentiable machine learning model for hydrology",
    author="Leo Lonzarich",
    author_email="lglonzaric@gmail.com",
    url="https://github.com/mhpi/dHBV_2_0",
    include_package_data=True,
    packages=find_packages(include=['dhbv_2_0', 'src',  'src.*']),
    # xarray==0.16.0 does not pin numpy, therefore transitively we pin numpy~=1.0
    # see https://github.com/NOAA-OWP/lstm/issues/46 for more detail.
    install_requires=["numpy~=1.0", "pandas", "bmipy", "torch", "pyyml", "netCDF4", "xarray==0.16.0"]
)
