# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""

# Configuration values for Rubin LSST simulations. Includes LSST instrument limits, magnitude depths, time range, and observational parameters.

# Saturation limits (magnitudes)
SATURATION_LIMITS = {
    'u': 14.7,
    'g': 15.7,
    'r': 15.8,
    'i': 15.8,
    'z': 15.3,
    'y': 13.9,
}

# 5-sigma depth (magnitudes)
FIVE_SIGMA_DEPTH = {
    'u': 23.78,
    'g': 24.81,
    'r': 24.35,
    'i': 23.92,
    'z': 23.34,
    'y': 22.45,
}

# Time range for simulation (MJD)
MJD_RANGE = {
    'min': 0,
    'max': 70000
}

PHOTO_PARAMS = {
    'exptime': 15,
    'nexp': 2
}

