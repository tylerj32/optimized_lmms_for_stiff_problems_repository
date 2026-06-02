import numpy as np
import matplotlib.pyplot as plt
import BDF_functions as bf
# In this file, we intend to run a convergence test on BDF-like methods to ensure 2nd
# order accuracy
h = np.logspace(-3, -1, 10) # h = time step
t0 = 1.5
lam = bf.lam1
def f(tn,un):
    return lam*(un-np.cos(tn))-np.sin(tn)
# Error lists
bdf2_err = np.zeros_like(h)
bdfl3_err = np.zeros_like(h)
bdfl4_err = np.zeros_like(h)
bdfl5_err = np.zeros_like(h)
# LMMs
for i in range(len(h)):
    t = np.arange(0, 6 + h[i], h[i])
    x_exact = np.exp(lam*t)/2 + np.cos(t)  # Exact Solution
    bdf2 = bf.LMM_solver(t,h[i],[1/3,-4/3,1],[0,0,2/3],t0,f)
    bdf2_err[i] = abs(bdf2[-1]-x_exact[-1])
    bdflike3 = bf.LMM_solver(t,h[i],[-1/10,3/5,-3/2,1],[0,0,0,3/5],t0,f)
    bdfl3_err[i] = abs(bdflike3[-1]-x_exact[-1])
    alpha4 = [1/41*(13-8*np.sqrt(2)),-2/41*(10-3*np.sqrt(2)),2/41*(1+12*np.sqrt(2)),
              -2/41*(18+11*np.sqrt(2)),1]
    beta4 = [0,0,0,0,4/41*(10-3*np.sqrt(2))]
    bdflike4 = bf.LMM_solver(t,h[i],alpha4,beta4,t0,f)
    bdfl4_err[i] = abs(bdflike4[-1]-x_exact[-1])
    alpha5 = [-1/88*(13-5*np.sqrt(5)),1/22*(10-3*np.sqrt(5)),-1/88*(25+9*np.sqrt(5)),
              np.sqrt(5)/2,-1/44*(45+14*np.sqrt(5)),1]
    beta5 = [0,0,0,0,0,5/44*(7-np.sqrt(5))]
    bdflike5 = bf.LMM_solver(t,h[i],alpha5,beta5,t0,f)
    bdfl5_err[i] = abs(bdflike5[-1]-x_exact[-1])
    print(f'Finished step {i+1} of {len(h)}')
plt.loglog(h,bdf2_err,label = 'bdf2')
plt.loglog(h,bdfl3_err, label = 'bdfl3')
plt.loglog(h,bdfl4_err, label = 'bdfl4')
plt.loglog(h,bdfl5_err, label = 'bdfl5')
plt.xlabel('h')
plt.ylabel('|$x_{numeric}-x_{exact}$|')
plt.grid(True, which='both')
plt.legend()
plt.show()
coeffs = np.polyfit(np.log(h), np.log(bdf2_err), 1)
print("Estimated order:", coeffs[0])