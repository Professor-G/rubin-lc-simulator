# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""


"""
Light curve models for use with Rubin/LSST simulations.

Includes:
- microlensing(): Simulates single-lens microlensing events
- constant(): Returns a flat baseline light curve
"""

import numpy as np


def constant(timestamps, baseline):
    """
    Simulate a constant (non-variable) light curve.

    Parameters
    ----------
    timestamps : array-like
        Times at which to simulate the light curve.
    baseline : float
        Constant baseline magnitude.

    Returns
    -------
    np.ndarray
        Magnitude array with constant value at each timestamp.
    """
    return np.full_like(timestamps, fill_value=baseline, dtype=np.float64)


def microlensing(timestamps, baseline, t0_dist=None, u0_dist=None, tE_dist=None):
    """
    Simulate a single-lens microlensing event.

    Parameters
    ----------
    timestamps : array-like
        Times at which to simulate the light curve.
    baseline : float
        Baseline magnitude of the light curve.
    t0_dist : tuple of float, optional
        Bounds for t0 (time of peak magnification). If None, estimated from timestamps.
    u0_dist : tuple of float, optional
        Bounds for u0 (minimum impact parameter). Default is (0.0, 1.0).
    tE_dist : tuple of float, optional
        Mean and stddev for tE (Einstein timescale). Default is N(30, 10) days.

    Returns
    -------
    microlensing_mag : np.ndarray
        Simulated magnitudes affected by microlensing.
    u_0 : float
        Minimum impact parameter.
    t_0 : float
        Time of peak magnification.
    t_e : float
        Event timescale (Einstein radius crossing time).
    blend_ratio : float
        Blend flux ratio (f_b / f_s).
    """

    # u0 parameter
    u0_min, u0_max = u0_dist if u0_dist else (0.0, 1.0)
    u_0 = np.random.uniform(u0_min, u0_max)

    # tE parameter
    tE_mean, tE_std = tE_dist if tE_dist else (30.0, 10.0)
    t_e = np.random.normal(tE_mean, tE_std)

    # t0 parameter
    if t0_dist:
        t0_min, t0_max = t0_dist
    else:
        t0_min = np.percentile(timestamps, 1) - 0.5 * t_e
        t0_max = np.percentile(timestamps, 99) + 0.5 * t_e
    t_0 = np.random.uniform(t0_min, t0_max)

    # Blend ratio (f_b / f_s)
    blend_ratio = np.random.uniform(0.0, 1.0)

    # Compute magnification
    u_t = np.sqrt(u_0**2 + ((timestamps - t_0) / t_e)**2)
    A_t = (u_t**2 + 2) / (u_t * np.sqrt(u_t**2 + 4))

    # Convert to magnitudes
    mag = constant(timestamps, baseline)
    flux = 10**(-0.4 * mag)
    f_s = np.median(flux) / (1 + blend_ratio)
    f_b = blend_ratio * f_s
    flux_obs = f_s * A_t + f_b
    microlensing_mag = -2.5 * np.log10(flux_obs)

    return microlensing_mag, u_0, t_0, t_e, blend_ratio
