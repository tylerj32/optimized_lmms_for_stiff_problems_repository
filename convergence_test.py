import numpy as np
import matplotlib.pyplot as plt
import BDF_functions as bf
import robertson_test as rt
# In this file, we intend to run a convergence test on BDF-like methods to ensure 2nd
# order accuracy
eps = 0.1
h = np.logspace(-3, -1, 10) # h = time
# t_1 *=t1
lam = -1e2
t0 = 1.5
def f(tn,un):
    return lam*(un-np.cos(tn))-np.sin(tn)
def jac(tn,un):
    return lam
# Error lists
bdf2_err = np.zeros_like(h)
bdfl3_err = np.zeros_like(h)
bdfl4_err = np.zeros_like(h)
bdfl5_err = np.zeros_like(h)
# LMMs
for i in range(len(h)):
    N = int(np.ceil(6/h[i]))  # roughly the same interval length
    n = np.arange(N)
    dt = h[i] * (1 + eps*np.sin(n*h[i]))
    t = np.concatenate(([0], np.cumsum(dt)))
    x_exact = np.exp(lam*t)/2 + np.cos(t)  # Exact Solution
    bdf2 = bf.LMM_solver2(t,rt.alpha_bdf2,rt.beta_bdf2,[t0],f,jac)
    bdf2_err[i] = abs(bdf2[-1,0]-x_exact[-1])
    bdflike3 = bf.LMM_solver2(t,rt.alpha_bdfl3_poly,rt.beta_bdfl3_poly,[t0],f,jac)
    bdfl3_err[i] = abs(bdflike3[-1,0]-x_exact[-1])
    bdflike4 = bf.LMM_solver2(t,rt.alpha_bdfl3,rt.beta_bdfl3,[t0],f,jac)
    bdfl4_err[i] = abs(bdflike4[-1,0]-x_exact[-1])
    bdflike5 = bf.LMM_solver2(t,rt.alpha_bdfl3_bdfopt,rt.beta_bdfl3_bdfopt,[t0],f,jac)
    bdfl5_err[i] = abs(bdflike5[-1,0]-x_exact[-1])
    print(f'Finished step {i+1} of {len(h)}')
plt.loglog(h,bdf2_err,label = 'bdf2')
plt.loglog(h,bdfl3_err, label = 'bdfl3-poly')
plt.loglog(h,bdfl4_err, label = 'bdfl3-order')
plt.loglog(h,bdfl5_err, label = 'bdfl3-bdfopt')
plt.xlabel('h')
plt.ylabel('|$x_{numeric}-x_{exact}$|')
plt.grid(True, which='both')
plt.legend()
plt.show()
coeffs = np.polyfit(np.log(h), np.log(bdf2_err), 1)
print("Estimated order (BDF2):", coeffs[0])
coeffs = np.polyfit(np.log(h), np.log(bdfl3_err), 1)
print("Estimated order (BDFL3-poly):", coeffs[0])
coeffs = np.polyfit(np.log(h), np.log(bdfl4_err), 1)
print("Estimated order (BDFL3-order):", coeffs[0])
coeffs = np.polyfit(np.log(h), np.log(bdfl5_err), 1)
print("Estimated order (BDFL3-bdfopt):", coeffs[0])