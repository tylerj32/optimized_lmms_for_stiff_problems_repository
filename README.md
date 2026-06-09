# optimized_lmms_for_stiff_problems_repository
The GitHub repository for the paper Optimized linear multistep methods for stiff problems
## Reproducing figures
# You will need python installed with libraries numpy, matplotlib and scipy
# BDFL and L(\kappa)-stable plots
# For this file, to get a comparison of the BDFL methods, comment out lines 63-108 and leave the rest as is
# To get a comparison of the L(\kappa)-stable methods, comment out lines 35-61
python BDF_like_comparison.py
# van der Pol (fixed step size) plots
python van_der_pol.py
# Robertson's problem plots
python robertson_test.py
# van der Pol (variable step size) plots
`python van_der_pol_variable_step.py`
