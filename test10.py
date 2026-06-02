import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import Testfuncs as tf
# BDFL3
# {-(1/10),3/5,-(3/2),1}
# {0,0,0,3/5}
# BDFL4
# {1/41 (13 - 8 Sqrt[2]), 2/41 (-10 + 3 Sqrt[2]),
#  2/41 (1 + 12 Sqrt[2]), -(2/41) (18 + 11 Sqrt[2]), 1}
#  {0,0,0,0,-(4/41) (-10 + 3 Sqrt[2])}
# BDFL5
# {1/88 (-13 + 5 Sqrt[5]), 1/22 (10 - 3 Sqrt[5]),
#  1/88 (-25 - 9 Sqrt[5]), Sqrt[5]/2, 1/44 (-45 - 14 Sqrt[5]), 1}
# {0,0,0,0,0,-(5/44) (-7 + Sqrt[5])}
# Initial conditions
a0 = (1/88)*(-13+5*np.sqrt(5))
a1 = (1/22)*(10-3*np.sqrt(5))
a2 = (1/88)*(-25-9*np.sqrt(5))
a3 = np.sqrt(5)/2
a4 = (1/44)*(-45-14*np.sqrt(5))
a5 = 1
b0 = 0
b1 = 0
b2 = 0
b3 = 0
b4 = 0
b5 = -(5/44)*(-7+np.sqrt(5))
# Order 0
eq0 = a0 + a1 + a2 + a3 + a4 + a5
# Order 1
eq1 = (a1 + 2*a2 + 3*a3 + 4*a4 + 5*a5) - (b0 + b1 + b2 + b3 + b4 + b5)
# Order 2
eq2 = (a1 + 4*a2 + 9*a3 + 16*a4 + 25*a5)/2 - (b1 + 2*b2 + 3*b3 + 4*b4 + 5*b5)
# Error constant for order 2
c3 = ((a1 + 8*a2 + 27*a3 + 64*a4 + 125*a5)/6 - (b1 + 4*b2 + 9*b3 + 16*b4 + 25*b5)/2)/(b0+b1+b2+b3+b4+b5)
print('0th order check: ',eq0)
print('1st order check: ',eq1)
print('2nd order check: ',eq2)
print('Error constant: ',c3)
a = [a0,a1,a2,a3,a4,a5]
b = [b0,b1,b2,b3,b4,b5]
a = np.array([float(x) for x in a])
b = np.array([float(x) for x in b])
z, check = tf.A_stable_test(a,b)
if not check:
    print('...and is therefore not L-stable')
else:
    check1, roots = tf.L_stable_test(a,b)
    if not check1:
        print('...but is not L-stable')
        print('Roots: ',roots)
    else:
        print('...and is L-stable')
        print('Roots: ',roots)
# Plotting
X, Y, stable = tf.stability_region_mask(a, b, xlim=(-0.005, 0.005), ylim=(-2, 2), N=300)
plt.figure()
plt.contourf(X, Y, stable, levels=[0.5, 1], colors=['lightblue'], alpha=0.6)
# Boundary locus
plt.plot(z.real, z.imag, 'k', linewidth=2)
plt.axvline(0, color='k', linestyle='--')
# plt.xlim(-3, 9)
# plt.ylim(-6, 6)
plt.xlim(-0.005,0.005)
plt.ylim(-2,2)
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Stability Region')
plt.grid()
plt.show()