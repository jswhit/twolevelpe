Python codes for ensemble data assimilation using a simple
two-level primitive equation spectral model on a sphere.


* twolevel.py:  model.
* enkf_utils.py: EnKF.
* pyspharm.py: Spherical harmonic routines (uses [shtns](https://bitbucket.org/nschaeff/shtns)).
* run_twolevel.py: generatre nature run.
* enkf_twolevel.py: run EnKF experiment.
* enkf_twolevel.py_iau.py: run EnKF experiment using 4D IAU.

1) To generate a nature run, execute ``python run_twolevel.py``.
2) To run an assimilation experiment, execute ``python enkf_twolevel.py <localization length scale> <inflation factor>``.
The localization length scale should be specified in meters (i.e. 4500.e3).  If the inflation factor give is between
0 and 1, relaxation to prior spread (RTPS) inflation is used.  If the inflation factor is exactly 1.0, then Hodyss
and Campbell's proposed inflation is used, with a=b=1. Global mean error and spread statistics for each analysis cycle
are written to stdout.

