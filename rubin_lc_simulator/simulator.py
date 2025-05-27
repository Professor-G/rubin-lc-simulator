# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:30:11 2018

@author: danielgodinez
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import numpy as np
import config  

import rubin_sim.maf as maf
from rubin_sim.phot_utils import signaltonoise, PhotometricParameters, rubin_bandpasses
from rubin_sim.data.rs_download_data import get_baseline


class LSSTSimulator:
    """
    Interface for simulating light curves using Rubin Observatory / LSST cadence and noise models.

    This class extracts the Rubin/LSST observational cadence at a specified sky position
    (RA, Dec) using the `rubin_sim` framework. It supports simulating photometric light curves
    by injecting user-defined models and applying realistic per-epoch noise based on
    the five sigma depth at each visit.

    Parameters
    ----------
    ra : float
        Right Ascension (in decimal degrees) of the simulated source. Default is 0.0.
    dec : float
        Declination (in decimal degrees) of the simulated source. Default is 0.0.
    band : str, optional
        LSST photometric filter to use ('u', 'g', 'r', 'i', 'z', or 'y'). Default is 'i'.
    out_dir : str, optional
        Output directory for storing cached metric results from `rubin_sim`. Default is '_metric_results_rubin_sim_'.

    Attributes
    ----------
    mjd_min : float
        Minimum MJD (time) for the simulation window.
    mjd_max : float
        Maximum MJD for the simulation window.
    photParams : PhotometricParameters
        LSST photometric configuration (e.g., exposure time, number of exposures).
    bandpasses : dict
        Dictionary of Rubin LSST bandpass filters from `rubin_sim.phot_utils`.
    opsim : Database
        Handle to the baseline LSST OpSim cadence simulation database.
    metric : Metric
        `rubin_sim` metric used to extract cadence and 5σ depth per observation.
    mjd : np.ndarray or None
        Simulated time values for the generated light curve.
    mag : np.ndarray or None
        Simulated magnitudes (with noise) of the light curve.
    magerr : np.ndarray or None
        Corresponding per-epoch magnitude uncertainties.
    """

    def __init__(
        self,
        ra: float = 0.0,
        dec: float = 0.0,
        band: str = "i",
        out_dir: str = "_metric_results_rubin_sim_",
    ) -> None:
        self.ra: float = ra
        self.dec: float = dec
        self.band: str = band.lower()
        self.out_dir: str = out_dir

        # Instrument limits & cadence settings (from config.py)
        self._m_sat: Dict[str, float] = config.SATURATION_LIMITS
        self._m_5_sigma: Dict[str, float] = config.FIVE_SIGMA_DEPTH
        self.mjd_min: float = config.MJD_RANGE["min"]
        self.mjd_max: float = config.MJD_RANGE["max"]

        # Photometric setup (from config.py)
        self.photParams: PhotometricParameters = PhotometricParameters(
            exptime=config.PHOTO_PARAMS["exptime"],
            nexp=config.PHOTO_PARAMS["nexp"],
        )
        self.bandpasses = rubin_bandpasses()

        # Baseline OpSim database handle
        self.opsim = get_baseline()

        # Metric for cadence extraction
        self.metric = maf.metrics.PassMetric(
            cols=["filter", "observationStartMJD", "fiveSigmaDepth"]
        )

        # Placeholders for a generated light-curve
        self.mjd: Optional[np.ndarray] = None
        self.mag: Optional[np.ndarray] = None
        self.magerr: Optional[np.ndarray] = None


    def __repr__(self) -> str:
        """Str representation of the class instance. """
        return f"<LSSTSimulator ra={self.ra:.3f}, dec={self.dec:.3f}, band='{self.band}'>"

    def _slice_sky(self) -> maf.slicers.UserPointsSlicer:
        """Slice the sky at the specified RA/DEC location.

        Returns:
            The spatial slicer the rubin_sim API uses to evaluate the sky on the spatial grid.
        """

        # The spatial slicer the rubin_sim API uses to evaluate the sky on the spatial grid.
        spatial_slicer = maf.slicers.UserPointsSlicer(ra=[self.ra], dec=[self.dec])
        
        return spatial_slicer

    def _retrieve_metrics(self) -> np.ndarray:
        """
        Query OpSim and return the raw metric data for the target position.

        Returns:
            The metrics data for the sky position. 
        """

        # Slice the sky at the specific location
        slicer = self._slice_sky() 

        # The bundle that will contain the metrics for this slice.
        bundleMetrics = maf.MetricBundle(
            self.metric, 
            slicer, 
            constraint='')

        # Extract the metrics using the run_all() class method. Setting to bundle variable in case I need to use later but not required atm
        bundle = maf.MetricBundleGroup([bundleMetrics], db_con=self.opsim, out_dir=self.out_dir).run_all()

        # The metric values for the MetricBundle.
        summary_metrics = bundleMetrics.metric_values[0]

        return summary_metrics 


    def LSST_metrics(self) -> Optional[np.ndarray]:
        """
        Public wrapper to obtain the visit table *without* any filtering and check if the slice is valid.

        Note:
            This method must always be run by the end-user, right after initialization. The spatial slicer this function returns
            containing the simulation metrics for the given sky position is never assigned as an attribute. This is a design choice. 
            This dataSlice must be assigned a local variable and managed by the user!

        Returns:
            The dataSlice which encapsulates the information on the visits overlapping a single point in the sky
        """

        # Retrieve the necessary metrics from rubin_sim
        data = self._retrieve_metrics()

        if isinstance(data, np.ma.core.MaskedConstant):
            print(f"WARNING: Empty data slice at RA={self.ra}, Dec={self.dec}")
            return None

        return data

    def _filter_band_time(self, data: np.ndarray) -> np.ndarray:
        """
        Restrict a data slice to the configured band and MJD range.

        Args:
            data (numpy.ndarray): Slice of data containing filter, observation start MJD, and five sigma depth.
        
        Returns:
            The filtered dataSlice containing the metrics.
        """

        mask = (
            (data["filter"] == self.band)
            & (data["observationStartMJD"] >= self.mjd_min)
            & (data["observationStartMJD"] <= self.mjd_max)
        )

        return data[mask]

    def _generate_light_curve(
        self, data: np.ndarray, lc: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Generate a light curve with Rubin cadence and LSST noise model.

        Note: 
            This method assigns local lightcurve variales only (e.g., mjd instead of self.mjd). 
            The lsst_real_lc() class method is the routine that assigns these as class attributes. 
            The code is structured this way as this method is intended to be run twice, the first time
            to extract the simulated timestamps, which the user can then use to simulate their lightcurve
            with their own model(s), after which the method can be re-run with the simulated magnitudes 
            during which the full lightcurve with appropriate errors can be constructed and saved.
        Args:
            data (np.ndarray): The dataSlice containing the cadence metrics for one position.
            lc (np.ndarray, optional): Ideal light curve. Can be a list or an array. Set to None when only the cadence/timestmaps are required.

        Returns:
            Either MJD array (if lc is None), or (mjd, mag, magerr) tuple.
        """

        mjd = data['observationStartMJD'] # Simulated cadence for the sky position

        # If no model is input (lc is None) simply return the scheduled cadence for the given sky position
        if lc is None:
            return mjd[np.argsort(mjd)] # Sorting always because I am paranoid!

        # If user inputs the magnitudes 
        if isinstance(lc, (np.ndarray, list)):

            # Mask the dataSlice to select the five sigman depth for the designated filter 
            filters = data['filter']
            five_sigma_depth = data['fiveSigmaDepth']

            # Using the the rubin_sim signaltonoise module, assign photometric errors to each data value using the simulated five sigma depths 
            magerr = np.array([
                signaltonoise.calc_mag_error_m5(lc[i], self.bandpasses[filters[i]], m5, self.photParams)[0]
                for i, m5 in enumerate(five_sigma_depth)
            ])

            # Errors are added to each point using a normal distribution as per the corresponding magerrs of each point. 
            mag = np.random.normal(loc=lc, scale=magerr)

            # Ensure data points are sorted according to the timestamps and return the full lightcurve
            sort = np.argsort(mjd)
            return mjd[sort], mag[sort], magerr[sort]
        else:
            raise ValueError('The inpuc `lc` must either be a list or an array!')

    def lsst_real_lc(
        self, data_slice: np.ndarray, lc: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Filter `data_slice` to the requested band and either return the cadence (MJD)
        or attach a full noisy light-curve to the instance.

        Args:
            data_slice : np.ndarray
                Output from `LSST_metrics()`.
            lc : np.ndarray | None, optional
                Ideal magnitudes to which LSST noise will be added.

        Returns
            np.ndarray | None
            Returns MJD array if lc=None, otherwise it does not return anything and instead assigns
            the following class attributes: (mjd, mag, magerr) tuple.
        """

        data_band = self._filter_band_time(data_slice)

        if lc is None:
            return self._generate_light_curve(data_band)

        if not isinstance(lc, (np.ndarray, list)):
            raise TypeError("`lc` must be a 1-D array-like of magnitudes.")

        self.mjd, self.mag, self.magerr = self._generate_light_curve(data_band, lc)

        return None


def draw_random_baseline(band: str) -> float:
    """
    Draw a random baseline magnitude within the allowed dynamic range.

    Parameters
    ----------
    band : str, optional
        LSST photometric filter to use ('u', 'g', 'r', 'i', 'z', or 'y'). 

    Returns
    -------
    float
        A baseline magnitude uniformly sampled between the saturation limit and 5σ depth for the specified band.
    """

    return np.random.uniform(
        config.SATURATION_LIMITS[band], config.FIVE_SIGMA_DEPTH[band]
    )



def draw_random_coord(ra_range=(0, 360), dec_range=(-75,15)):
    """
    Draw coordinates using randomly uniform distributions.

    Args:
        ra_range (tuple): Right ascension range to draw from, in degrees. Defaults to entire sky (0-360).
        dec_range (tuple): Declination range to draw from, in degrees. Defaults to (-70, 12), sampling over the Southern sky that LSST will observe

    Returns:
        List with two floats, ra and dec in decimal degrees
    """

    ra = np.random.uniform(ra_range[0], ra_range[1])
    dec = np.degrees(np.arcsin(np.random.uniform(np.sin(np.radians(dec_range[0])), np.sin(np.radians(dec_range[1])))))

    return [ra, dec]
