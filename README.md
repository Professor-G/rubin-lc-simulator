# rubin-lc-simulator
A lightweight Python package for simulating time-domain light curves with Rubin/LSST cadence and photometric uncertainty. Supports sky-position-dependent cadence extraction via rubin_sim, customizable light curve injection using the LSST Camera's six optical filters (ugrizy), and realistic noise modeling based on Rubin’s five sigma depth.

# Installation
The latest version can be installed via pip

```
    $ pip install rubin-lc-simulator
```

While the code requires only numpy, it is wrapped around the [rubin_sim API.](https://rubin-sim.lsst.io/)

This must be installed manually, please follow the instructions on their documentation. 

# [Documentation](https://rubin-lc-simulator.readthedocs.io/en/latest/index.html)

For technical details and an example of how to use the code, check out our [Documentation](https://rubin-lc-simulator.readthedocs.io/en/latest/index.html).

# Citation

If you use this simulation code in publication, we would appreciate citations to the paper, [Romao, Croon, & Godines 2025](https://arxiv.org/abs/2503.09699)

 
# How to Contribute?

Want to contribute? Bug detections? Comments? Suggestions? Please email us : danielgodinez123@gmail.com
