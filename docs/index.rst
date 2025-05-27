.. rubin-lc-simulator documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rubin-lc-simulator's documentation!
===============================
This is an open-source program for simulating light curves with realistic LSST cadence and photometric noise. The code is designed to work with any given light curve model, although for testing purposes we provide a function for simulating microlensing events (PSPL only) and constant, signal-less lightcurves.

This framework was designed and used for research in anomaly detection techniques, (Romao, Croon, & Godines 2025). If you use this code for your own research we would appreciate citations to `our paper <https://arxiv.org/abs/2503.09699>`_.

Installation
==================
The current stable version can be installed via pip:

.. code-block:: bash

    pip install rubin-lc-simulator


The code utilizes the `rubin_sim <https://rubin-sim.lsst.io/>`_ Python package. Please follow the installation instructions on their `documentation page <https://rubin-sim.lsst.io/installation.html#quick-installation>`_ and ensure the following imports work before using this code:

.. code-block:: python

   from rubin_sim import maf, phot_utils, data


Pages
==================
.. toctree::
   :maxdepth: 1

   source/Example

Documentation
==================
Here is the documentation for all the modules:

.. toctree::
   :maxdepth: 1

   source/rubin_lc_simulator
