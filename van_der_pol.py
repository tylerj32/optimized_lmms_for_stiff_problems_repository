import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import hilbert
import BDF_functions as bf
import BDF_like_comparison as blc
from scipy.integrate import solve_ivp
# In this script, we will attempt to solve a system of ODEs defined below:
# dy_1/dt = y2
# dy_2/dt = \mu*(1-y_1^2)y2-y1
# This denotes a self oscillating system that dissipates. It becomes very stiff as \mu = 500
t0 = [2.0, 0.0]
h = .001
# 2000
t_2 = np.arange(0, 2000+h, h)
mu = 500
def f(t, u):
    y1, y2 = u
    return np.array([
        y2,
        mu*(1 - y1**2)*y2 - y1
    ])
def jac(t, u):
    y1, y2 = u
    return np.array([
        [0.0,  1.0],
        [2*mu*y2*y1-1.0,   mu*(1-y1**2)]
    ])
# BDFL methods
bdf2 = bf.LMM_solver2(t_2,[1/3,-4/3,1],[0,0,2/3],t0,f,jac)
bdflike3 = bf.LMM_solver2(t_2,[-1/10,3/5,-3/2,1],[0,0,0,3/5],t0,f,jac)
bdflike4 = bf.LMM_solver2(t_2,blc.alpha4,blc.beta4,t0,f,jac)
bdflike5 = bf.LMM_solver2(t_2,blc.alpha5,blc.beta5,t0,f,jac)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sol = np.array([bdf2,bdflike3,bdflike4,bdflike5])
ref = solve_ivp(f, [t_2[0], t_2[-1]], t0, method='Radau', jac=jac, rtol=1e-10, atol=1e-12, dense_output=True)
ref_vals = ref.sol(t_2[:-1])  # interpolated solution at t_2
line1 = blc.line
method_handles = []
plt.plot(t_2[:-1],ref_vals[0], color='black', lw=2, label='Exact solution')
method_handles.append(
        Line2D([0], [0], color='black', lw=2, label='Exact solution')
    )
for i in range(len(sol)):
    y1 = sol[i,:-1,0]
    label = 'BDF2' if i==0 else f'BDFL${i+2}_2$'
    method_handles.append(
        Line2D([0], [0], color=default_colors[i+1], lw=2, label=label,linestyle=line1[i])
    )
    plt.plot(t_2[:-1], y1, color=default_colors[i+1], label=label, linestyle=line1[i])
legend1 = plt.legend(handles=method_handles, title="Method",loc='lower left')
plt.xlabel('t')
plt.ylabel(r'$y_2$',rotation=0)
plt.title('Numerical Result - van der Pol')
plt.tight_layout()
plt.grid()
plt.show()
# Error plot
y2_exact = ref_vals[1]
analytic_exact = hilbert(y2_exact)
phase_exact = np.unwrap(np.angle(analytic_exact))
phase_diff_set = []
for s in sol:
    # Use y2 component
    y2_num = s[:-1, 1]
    # Performing Hilbert transform
    analytic_sol = hilbert(y2_num)
    # Calculating phases
    phase_sol = np.unwrap(np.angle(analytic_sol))
    # Phase difference
    phase_diff_sol = phase_sol - phase_exact
    phase_diff_set.append(np.abs(phase_diff_sol))
for i in range(len(phase_diff_set)):
    label = 'BDF2' if i == 0 else f'BDFL${i+2}_2$'
    plt.semilogy(t_2[:-1], phase_diff_set[i], color=default_colors[i+1],linestyle = line1[i],label = label)
plt.ylabel(r'$|\text{Phase Error}|$')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
