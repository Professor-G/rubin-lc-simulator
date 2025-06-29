# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike

def constant(timestamps: ArrayLike, baseline: float) -> np.ndarray:
    """
    Generate a non-variable light curve.

    Parameters
    ----------
    timestamps : array-like
        Observation epochs (any shape) in days.
    baseline : float
        Constant magnitude to assign.

    Returns
    -------
    np.ndarray
        Array with the same shape as `timestamps`, filled with `baseline`.
    """
    return np.full_like(np.asarray(timestamps, dtype=float),
                        fill_value=baseline,
                        dtype=float)


def microlensing(
    timestamps: ArrayLike,
    baseline: float,
    t0_dist: Tuple[float, float] | None = None,
    u0_dist: Tuple[float, float] | None = None,
    tE_dist: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Simulate a single-lens, point-source microlensing event with blending.

    If not defined, the microlensing parameter space is determined using data from an 
    analysis of the OGLE III microlensing survey from Y. Tsapras et al (2016).
    See: The OGLE-III planet detection efficiency from six years of microlensing observations (2003 to 2008).
    (https://arxiv.org/abs/1602.02519)

    Parameters
    ----------
    timestamps : array-like
        Observation epochs (days).
    baseline : float
        Baseline magnitude outside the event.
    t0_dist : (float, float), optional
        Uniform bounds for ``t0`` (time of peak).  Defaults to the middle
        98% of the supplied `timestamps` plus a cushion of 0.5 `t_E` on each
        side.
    u0_dist : (float, float), optional
        Uniform bounds for ``u0`` (impact parameter).  Default: (0, 1).
    tE_dist : (float, float), optional
        Mean and standard deviation for normally distributed ``t_E``
        (Einstein-radius crossing time).  Default: 30 plus/minus 10 days.

    Returns
    -------
    Tuple
        * `mag` – simulated magnitudes (np.ndarray)
        * `u0` – drawn impact parameter
        * `t0` – drawn peak time (days)
        * `tE` – drawn event timescale (days)
        * `blend_ratio` – flux blending ratio *f_b / f_s*
    """
    
    ts = np.asarray(timestamps, dtype=float)

    # Draw parameters
    u0_min, u0_max = u0_dist if u0_dist is not None else (0.0, 1.0)
    u0 = np.random.uniform(u0_min, u0_max)

    tE_mean, tE_std = tE_dist if tE_dist is not None else (30.0, 10.0)
    tE = np.random.normal(tE_mean, tE_std)

    if t0_dist is not None:
        t0_min, t0_max = t0_dist
    else:
        t0_min = np.percentile(ts, 1) - 0.5 * tE
        t0_max = np.percentile(ts, 99) + 0.5 * tE
    t0 = np.random.uniform(t0_min, t0_max)

    blend_ratio = np.random.uniform(0.0, 1.0)

    # Magnification curve
    u_t = np.sqrt(u0**2 + ((ts - t0) / tE) ** 2)
    A_t = (u_t**2 + 2.0) / (u_t * np.sqrt(u_t**2 + 4.0))

    # Convert to observed magnitudes with blending 
    mag_base = constant(ts, baseline)
    flux_base = 10.0 ** (-0.4 * mag_base)

    f_s = np.median(flux_base) / (1.0 + blend_ratio)  # source flux
    f_b = blend_ratio * f_s # blend flux
    flux_obs = f_s * A_t + f_b

    mag_obs = -2.5 * np.log10(flux_obs)

    return mag_obs, u0, t0, tE, blend_ratio
