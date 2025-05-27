# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:35:19 2025

@author: danielgodinez
"""

from setuptools import setup, find_packages

setup(
    name="rubin-lc-simulator",
    version="1.11",
    author="Daniel Godines",
    author_email="danielgodinez123@gmail.com",
    description="Open-source tool to simulate LSST light curves with Rubin cadence and per-epoch noise.",
    license='MIT',
    url="https://github.com/Professor-G/rubin-lc-simulator",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    packages=find_packages('.'),
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
