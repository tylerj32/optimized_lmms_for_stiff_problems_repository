import numpy as np
import matplotlib.pyplot as plt
import BDF_functions as bf
# This file will generate and compare the BDF-like LMMs to common methods such as BDF2 and
# the Trapezoidal Rule. For that reason, our differential equation will be extremely stiff
# to ensure that the accurate results we are getting only come from L-stable LMMs

# Our example of a stiff equation is given in LeVeque, page 167:
# u'(t) = lambda(u-cos(t))-sin(t)
# The same page offers the analytical result, and page 172 offers values for lambda, t0, and k.
# We will take these to try to recreate the figures at the top of page 173
k = bf.k1 # tau = time step
t = bf.t1
lam = bf.lam1
tan = bf.tan
x_exact = bf.x_exact # Exact Solution
x_an = bf.x_an # A finer version of the exact solution, which will be
# put on the graph
x_num, x_num1 = bf.lmm1(t,k,lam)
# Comparison plot
plt.plot(tan,x_an, label='Analytical Solution')
plt.plot(t,x_num, label='Numerical Solution (Trapezoidal)')
plt.plot(t,x_num1, label='Numerical Solution (Implicit Euler)')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.show()
# Difference plot
plt.semilogy(t,np.abs(x_num-x_exact), label='Trapezoidal')
plt.semilogy(t,np.abs(x_num1-x_exact), label='Implicit Euler')
plt.title('Difference between Numerical Solution and Analytical Solution')
plt.xlabel('t')
plt.ylabel('|$x_{numeric}-x_{exact}$|')
plt.grid()
plt.legend()
plt.show()