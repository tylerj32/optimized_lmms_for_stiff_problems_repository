import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import BDF_functions as bf
import BDF_like_comparison as blc
# In this script, we will attempt to solve a system of ODEs defined below:
# dy_1/dt = -K1*y1 + 0*y2 + 0*y3,
# dy_2/dt = K1*y1 - K2*y2 + 0*y3,
# dy_3/dt = 0*y1 + K2*y2 + 0*y3
# This denotes a chemical reaction where y_1 (A) decays into y_2 (B), which in turn decays into y_3 (C)
t0 = [3.0, 4.0, 2.0]
h = 1e-1
t_1 = np.arange(0, 8+h, h)
K1 = 3
K2 = 1
def f(t, u):
    y1, y2, y3 = u
    return np.array([
        -K1*y1,
        K1*y1 - K2*y2,
        K2*y2
    ])
def jac(t, u):
    y1, y2, y3 = u
    return np.array([
        [-K1,  0.0,    0.0],
        [K1,   -K2,    0.0],
        [0.0,  -K2,    0.0]
    ])
# BDFL methods
bdf2 = bf.LMM_solver2(t_1,[1/3,-4/3,1],[0,0,2/3],t0,f,jac)
bdflike3 = bf.LMM_solver2(t_1,[-1/10,3/5,-3/2,1],[0,0,0,3/5],t0,f,jac)
bdflike4 = bf.LMM_solver2(t_1,blc.alpha4,blc.beta4,t0,f,jac)
bdflike5 = bf.LMM_solver2(t_1,blc.alpha5,blc.beta5,t0,f,jac)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sol = np.array([bdf2,bdflike3,bdflike4,bdflike5])
method_handles = []
for i in range(len(sol)):
    y1 = sol[i, :, 0]
    y2 = sol[i,:,1]
    y3 = sol[i,:,2]
    label = 'BDF2' if i==0 else f'BDFL{i+2}'
    method_handles.append(
        Line2D([0], [0], color=default_colors[i], lw=2, label=label)
    )
    plt.plot(t_1, y1, color=default_colors[i], label=label, linestyle='-')
    plt.plot(t_1, y2*1e4, color=default_colors[i], linestyle='--')
    plt.plot(t_1, y3, color=default_colors[i], linestyle=':')
legend1 = plt.legend(handles=method_handles, title="Method", loc='upper right')
plt.gca().add_artist(legend1)  # keep this legend when adding another

# Second legend: variables (linestyles)
style_handles = [
    Line2D([0], [0], color=default_colors[3], lw=2, linestyle='-',  label=r'$y_1$'),
    Line2D([0], [0], color=default_colors[3], lw=2, linestyle='--', label=r'$y_2 \times 10^4$'),
    Line2D([0], [0], color=default_colors[3], lw=2, linestyle=':',  label=r'$y_3$'),
]

plt.legend(handles=style_handles, title="Chemical compound", loc='right')
plt.xlabel('t')
plt.ylabel('Concentration')
plt.title(r'Chemical decay of $y_1 \to y_2 \to y_3$')
plt.tight_layout()
plt.grid()
plt.show()
# Error plot
y1_exact = 3*np.exp(-3*t_1)
y2_exact = 1/2*np.exp(-3*t_1)*(17*np.exp(2*t_1)-9)
y3_exact = 9 + (3*np.exp(-3*t_1))/2 - (17*np.exp(-t_1))/2
fig, axs = plt.subplots(3, 1, sharex=True)  # 3 rows, 1 column
for i in range(len(sol)):
    y1 = sol[i,:,0]
    y2 = sol[i,:,1]
    y3 = sol[i,:,2]
    label = 'BDF2' if i == 0 else f'BDFL{i+2}'
    axs[0].semilogy(t_1, np.abs(y1-y1_exact), color=default_colors[i], label=label)
    axs[1].semilogy(t_1, np.abs(y2-y2_exact), color=default_colors[i])
    axs[2].semilogy(t_1, np.abs(y3-y3_exact), color=default_colors[i])
# Labels for each subplot
axs[0].set_ylabel(r'$y_1$ - Error',rotation=0, labelpad=30)
axs[1].set_ylabel(r'$y_2$ - Error',rotation=0, labelpad=30)
axs[2].set_ylabel(r'$y_3$ - Error',rotation=0, labelpad=30)
axs[2].set_xlabel('t')

# Only one legend (usually top plot)
axs[0].legend(loc=1)

plt.tight_layout()
plt.show()
