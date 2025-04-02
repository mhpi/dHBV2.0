# TODO: Replace with the pyproject.toml once we guarantee everything is working in ngen.


#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='dHBV_2_0',
    version='1.0.0',
    description="High-resolution, multiscale, differentiable machine learning model for hydrology",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Leo Lonzarich',
    author_email='lglonzaric@gmail.com',
    url='https://github.com/mhpi/dHBV_2_0',
    license='Apache-2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache License 2.0',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    packages=find_packages(include=['src']),
    # xarray==0.16.0 does not pin numpy, therefore transitively we pin numpy~=1.0
    # see https://github.com/NOAA-OWP/lstm/issues/46 for more detail.
    install_requires=[
        'numpy~=1.0',
        'pandas',
        'bmipy',
        'torch',
        'pyyml',
        'netCDF4',
        'h5netcdf',
        'xarray==0.16.0',
    ],
    python_requires='>=3.9',
)
