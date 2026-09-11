# optimized_lmms_for_stiff_problems_repository
The GitHub repository for the paper Optimized Linear Multistep Methods for Stiff Problems
## Reproducing figures
# You will need python installed with libraries numpy, matplotlib and scipy
# BDFL and L(\kappa)-stable plots. For this file, to get a comparison of the BDFL methods, set s=0 on line 28. To get a comparison of the L(\kappa)-stable methods, set s=1 on line 28
`python BDF_like_comparison.py`
# van der Pol (fixed step size) plots
`python van_der_pol.py`
# Robertson's problem plots
`python robertson_test.py`
# van der Pol (variable step size) plots
`python van_der_pol_variable_step.py`
# Python functions necessary for all of the .py files
`python BDF_functions.py`
# Preliminary tests for the BDFL methods
`python BDF_like_test.py`
# Convergence tests on the variable step size BDFL methods
`python convergence_test.py`
# Test file on a system of ODEs for the BDFL methods
`python system_of_ODEs_test_bdfl.py`
# RLC curves of the BDFL methods
`python test10.py`
# Python functions for the test10.py file
`python Testfuncs.py`
## Mathematica
# A-stability for 2-step 2nd order methods
`wolframscript -file 2step_LMM_Conditions.nb`
# The optimal BDFL4_2 and BDFL5_2 methods
`wolframscript -file 4_and_5step_Lstable.nb`
# An analysis of the BDF2OPT formula from Carpenter et. al (2010), where we replicate their work and discover that the BDFL3_2 and BDFL4_2 methods match BDF2OPT(4) and BDF2OPT(5) formulas
`wolframscript -file BDF2OPT_analysis_3.nb`
# Initial trials in setting up the set of optimal L(\kappa)-stable methods
`wolframscript -file k_to_c.nb`
# Final results for the optimal L(\kappa)-stable methods via numerical methods
`wolframscript -file k_to_c_numerical.nb`
# A set of 100 coefficients, from \kappa=0 to 1, with the smallest error constant modulus (corresponds to the black dotted line in Figure 6)
`wolframscript -file k_to_c_coeffs.nb`
# Deriving the consistency conditions for variable step size methods
`wolframscript -file variable_step_conditions.nb`
# Deriving the variable step BDFL methods via the polynomial method and by the order conditions (also checking for 0-stability)
`wolframscript -file variable_step_coeffs.nb`
