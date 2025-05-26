import numpy as np
import rubin_sim.maf as maf
from rubin_sim.phot_utils import signaltonoise, PhotometricParameters, rubin_bandpasses
from rubin_sim.data.rs_download_data import get_baseline

#############
# 1) Below is the routine I wrote to simulate lightcurves using the rubin_sim package, it extracts the cadence at a given location in the sky, and adds noise using rubin_sim.phot_utils.signaltonoise
#############


class LSSTSimulator:
    def __init__(self, ra=0, dec=0, band='i', out_dir="_metric_results_rubin_sim_"):
        """
        Simulate light curves using Rubin cadence and noise for a given band and sky position.
        """
        self.ra = ra
        self.dec = dec
        self.band = band
        self.out_dir = out_dir

        # LSST instrument specs
        self.m_sat = {'u': 14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
        self.m_5_sigma = {'u': 23.78, 'g': 24.81, 'r': 24.35, 'i': 23.92, 'z': 23.34, 'y': 22.45}
        self.mjd_min = 0
        self.mjd_max = 61400

        # Photometric parameters for LSST (30s exposures)
        self.photParams = PhotometricParameters(exptime=15, nexp=2)
        self.LSST_BandPass = rubin_bandpasses()
        self.opsim = get_baseline()

        # Metric: get cadence and depth
        self.metric = maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth'])

    def __repr__(self):
        """Str representation of the class instance. """

        return f"<LSSTSimulator ra={self.ra}, dec={self.dec}, band='{self.band}'>"

    def random_baseline(self):
        """Helper function to randomly choose a baseline magnitude according to the LSST saturation limits and 5 sigma depths for the given filter.
        
        Returns:
            Random baseline float within the telescope observational limits, for the configured filter
        """

        return np.random_uniform(self.m_sat[self.band], self.m_5_sigma[self.band])

    def slice_sky(self):
        """Slice the sky at the specified RA/DEC location.

        Returns:
            The spatial slicer the rubin_sim API uses to evaluate the sky on the spatial grid.
        """

        spatial_slicer = maf.slicers.UserPointsSlicer(ra=[self.ra], dec=[self.dec])# The spatial slicer the rubin_sim API uses to evaluate the sky on the spatial grid.
        
        return spatial_slicer

    def retrieve_metrics(self):
        """Retrieve observing cadence and depth from the rubin_sim API.

        Returms:
            The metrics data for the sky position. 
        """

        slicer = self.slice_sky() # Slice the sky at the specific location

        # The bundle that will contain the metrics for this slice.
        bundleMetrics = maf.MetricBundle(self.metric, slicer, constraint='')

        # Extract the metrics using the run_all() class method. Setting to bundle variable in case I need to use later but not required atm
        bundle = maf.MetricBundleGroup([bundleMetrics], db_con=self.opsim, out_dir=self.out_dir).run_all()

        # The metric values for the MetricBundle.
        summary_metrics = bundleMetrics.metric_values[0]

        return summary_metrics 

    def LSST_metrics(self):
        """Retrieve observing metrics and check if the slice is valid.

        Note:
            This method must always be run by the end-user, right after initialization. The spatial slicer this function returns
            containing the simulation metrics for the given sky position is never assigned as an attribute. This is a design choice. 
            This dataSlice must be assigned a local variable and managed by the user!

        Returns:
            The dataSlice which encapsulates the information on the visits overlapping a single point in the sky
        """

        dataSlice = self.retrieve_metrics() # Retrieve the necessary metrics from rubin_sim

        if isinstance(dataSlice, np.ma.core.MaskedConstant):
            print(f"WARNING: Empty data slice at RA={self.ra}, Dec={self.dec}")
            return None

        return dataSlice

    def filter_data(self, dataSlice):
        """Filter data by band and MJD range.

        Args:
            dataSlice (numpy.ndarray): Slice of data containing filter, observation start MJD, and five sigma depth.
        
        Returns:
            The filtered dataSlice containing the metrics.
        """

        mask = (
            (dataSlice['filter'] == self.band.lower()) &
            (dataSlice['observationStartMJD'] >= self.mjd_min) &
            (dataSlice['observationStartMJD'] <= self.mjd_max)
        )

        return dataSlice[mask]

    def generate_light_curve(self, dataSlice, lc=None):
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
            dataSlice (np.ndarray): Rubin cadence metrics for one position.
            lc (np.ndarray, optional): Ideal light curve (same length as dataSlice). 
                Can be a list or an array. Set to None when only the cadence/timestmaps are required.

        Returns:
            Either MJD array (if lc is None), or (mjd, mag, magerr) tuple.
        """

        mjd = dataSlice['observationStartMJD'] # Simulated cadence for the sky position

        # If user inputs the magnitudes 
        if isinstance(lc, (np.ndarray, list)):

            # Mask the dataSlice to select the five sigman depth for the designated filter 
            filters = dataSlice['filter']
            five_sigma_depth = dataSlice['fiveSigmaDepth']

            # Using the the rubin_sim signaltonoise module, assign photometric errors to each data value using the simulated five sigma depths 
            magerr = np.array([
                signaltonoise.calc_mag_error_m5(lc[i], self.LSST_BandPass[filters[i]], m5, self.photParams)[0]
                for i, m5 in enumerate(five_sigma_depth)
            ])

            # Errors are added to each point using a normal distribution as per the corresponding magerrs of each point. 
            mag = np.random.normal(loc=lc, scale=magerr)

            # Ensure data points are sorted according to the timestamps and return the full lightcurve
            sort = np.argsort(mjd)
            return mjd[sort], mag[sort], magerr[sort]

        # If no model is input (lc is None) simply return the scheduled cadence for the given sky position
        return mjd[np.argsort(mjd)] # Sorting always because I am paranoid!

    def lsst_real_lc(self, dataSlice, lc=None):
        """
        Filter data and generate a noise-perturbed light curve (or return MJD).
        This class method assigns the lightcurve data as class attributes (e.g., self.mjd)

        Args:
            dataSlice (np.ndarray): Rubin cadence metrics.
            lc (np.ndarray, optional): Ideal light curve values.

        Returns:
            Returns MJD array if lc=None, otherwise it does not return anything and instead assigns
            the following class attributes: (mjd, mag, magerr) tuple.
        """

        filtered_data = self.filter_data(dataSlice)

        # If no magnitudes (lc) is input, then simply return the simulated timestamps
        if lc is None:
            return self.generate_light_curve(filtered_data, lc)

        # If lc is input, then assign the entire lightcurve (mjd, mag, magerr) as class attributes
        elif isinstance(lc, (np.ndarray, list)):
            self.mjd, self.mag, self.magerr = self.generate_light_curve(filtered_data, lc)

        else:
            print('ERROR: lc must be a 1D list or array containing the ideal (i.e., error-free) magnitudes from your model.')

        return 


