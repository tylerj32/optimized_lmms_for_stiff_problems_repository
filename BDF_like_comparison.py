import numpy as np
import matplotlib.pyplot as plt
import BDF_functions as bf
# from scipy import integrate as int
# # IVP Solver
# def exponential_decay(t, y): return lam*(y-np.cos(t))-np.sin(t)
# sol = int.solve_ivp(exponential_decay, [0, 3], [1.5])
# plt.plot(sol.t, sol.y[0], label='Numerical Solution (IVP)')

# In this script, we will take the classic BDF2 and compare its performance to BDF-like 3, 4, and
# 5-step schemes. See BDF_like_test for an explanation of the equation used for this performance.
h = bf.h1 # h = time step
t0 = 1.5
lam = bf.lam1
def f(tn,un):
    return lam*(un-np.cos(tn))-np.sin(tn)
def f_comp(tn,un):
    return -np.sin(tn)
# BDFL4 and 5 coefficients
alpha4 = [1/41*(13-8*np.sqrt(2)),-2/41*(10-3*np.sqrt(2)),2/41*(1+12*np.sqrt(2)),
          -2/41*(18+11*np.sqrt(2)),1]
beta4 = [0,0,0,0,4/41*(10-3*np.sqrt(2))]
alpha5 = [-1/88*(13-5*np.sqrt(5)),1/22*(10-3*np.sqrt(5)),-1/88*(25+9*np.sqrt(5)),
              np.sqrt(5)/2,-1/44*(45+14*np.sqrt(5)),1]
beta5 = [0,0,0,0,0,5/44*(7-np.sqrt(5))]
line = ['-', '--', ':', '-.', (0, (3, 1,1,1))]
if __name__ == "__main__":
    s=0
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Extracts default color scheme for plots. This
    # will be used to make the colors in the second plot match the lines in the first
    if s==0:
        t = np.arange(0, 3 + h, h)
        # n-cos(t0) => 1.5-1
        tan = np.arange(0, 3, .01)
        y_exact = np.exp(lam * t) / 2 + np.cos(t)  # Exact Solution
        y_an = np.exp(lam * tan) / 2 + np.cos(tan)
        # BDF-like
        bdf2 = bf.LMM_solver(t, h, [1 / 3, -4 / 3, 1], [0, 0, 2 / 3], t0, f)
        bdflike3 = bf.LMM_solver(t, h, [-1 / 10, 3 / 5, -3 / 2, 1], [0, 0, 0, 3 / 5], t0, f)
        bdflike4 = bf.LMM_solver(t,h,alpha4,beta4,t0,f)
        bdflike5 = bf.LMM_solver(t,h,alpha5,beta5,t0,f)
        # Comparison plot
        plt.plot(tan,y_an,label='Exact Solution')
        plt.plot(t,bdf2, label='BDF2',linestyle=line[0])
        plt.plot(t,bdflike3, label=r'BDFL$3_2$',linestyle=line[2])
        plt.plot(t,bdflike4, label=r'BDFL$4_2$',linestyle=line[3])
        plt.plot(t,bdflike5, label=r'BDFL$5_2$',linestyle=line[4])
        plt.ylabel('y(t)',rotation=0)
        plt.xlabel('t')
        plt.xlim([t[0],t[-1]])
        plt.legend()
        plt.grid()
        plt.show()
        # BDFL Difference plot
        plt.semilogy(t,np.abs(bdf2-y_exact), label='BDF2', color=default_colors[1], linestyle=line[0])
        plt.plot(t,np.abs(bdflike3-y_exact), label=r'BDFL$3_2$', color=default_colors[2], linestyle=line[1])
        plt.plot(t,np.abs(bdflike4-y_exact), label=r'BDFL$4_2$', color=default_colors[3], linestyle=line[2])
        plt.plot(t,np.abs(bdflike5-y_exact), label=r'BDFL$5_2$', color=default_colors[4], linestyle=line[3])
        plt.ylabel(r'$|y_{\text{exact}}-y_{\text{numeric}}|$')
        plt.xlabel('t')
        plt.xlim([t[0], t[-1]])
        plt.legend()
        plt.grid()
        plt.show()
    else:
        t = np.arange(0, 6 + h, h)
        # n-cos(t0) => 1.5-1
        tan = np.arange(0, 6, .01)
        y_exact = np.exp(lam * t) / 2 + np.cos(t)  # Exact Solution
        y_an = np.exp(lam * tan) / 2 + np.cos(tan)
        # L(kappa)-stable
        # Trapezoidal
        trap = bf.LMM_solver(t, h, [-1, 1], [1 / 2, 1 / 2], t0, f)
        # plt.plot(tan,y_an,label='Exact Solution')
        # plt.grid()
        # plt.legend()
        # plt.show()
        # k = 1/5
        # kappa15 = bf.LMM_solver(t,h,[-(5/69), 31/69, -(95/69), 1],[0, 0, 8/69, 40/69],t0,f)
        # k = 2/5
        kappa25 = bf.LMM_solver(t,h,[-(15/304), 6/19, -(385/304), 1],[0, 0, 17/76, 85/152],t0,f)
        # k = 3/5
        kappa35 = bf.LMM_solver(t,h,[-(1/3), -(2/3), 1],[3/16, 5/8, 25/48],t0,f)
        # k = 4/5
        kappa45 = bf.LMM_solver(t,h,[-(7/11), -(4/11), 1],[32/99, 80/99, 50/99],t0,f)
        fig, axs = plt.subplots(2, 2, sharex=False)  # 2 rows, 2 columns
        axs = axs.flatten()  # makes indexing easier
        axs[0].plot(tan, y_an, linestyle=line[0], label='Exact Solution')
        axs[0].plot(t, kappa25, linestyle=line[1], color=default_colors[2], label=r'$\kappa = \frac{2}{5}$')
        axs[1].plot(tan, y_an, linestyle=line[0], label='Exact Solution')
        axs[1].plot(t, kappa35, linestyle=line[2], color=default_colors[3], label=r'$\kappa = \frac{3}{5}$')
        axs[2].plot(tan, y_an, linestyle=line[0], label='Exact Solution')
        axs[2].plot(t, kappa45, linestyle=line[3], color=default_colors[4], label=r'$\kappa = \frac{4}{5}$')
        axs[3].plot(tan, y_an, linestyle=line[0], label='Exact Solution')
        axs[3].plot(t, trap, linestyle=line[4], color=default_colors[5], label=r'Trapezoidal rule ($\kappa = 1$)')
        for ax in axs:
            ax.set_xlabel('t')
            ax.set_ylabel('y(t)',rotation=0)
            ax.set_xlim([t[0], t[-1]])
            ax.grid()
            ax.legend(loc=1)
        plt.tight_layout()
        plt.show()
        # L(k)-stable error plots
        bdf2 = bf.LMM_solver(t, h, [1 / 3, -4 / 3, 1], [0, 0, 2 / 3], t0, f)
        plt.semilogy(t, np.abs(bdf2 - y_exact), label='BDF2', color=default_colors[1], linestyle=line[0])
        plt.plot(t,np.abs(kappa25-y_exact), label=r'$\kappa = \frac{2}{5}$', color=default_colors[2], linestyle=line[1])
        plt.plot(t,np.abs(kappa35-y_exact), label=r'$\kappa = \frac{3}{5}$', color=default_colors[3], linestyle=line[2])
        plt.plot(t,np.abs(kappa45-y_exact), label=r'$\kappa = \frac{4}{5}$', color=default_colors[4], linestyle=line[3])
        plt.plot(t,np.abs(trap-y_exact), label=r'Trapezoidal rule ($\kappa = 1$)', color=default_colors[5], linestyle=line[4])
        plt.title('Difference between Numerical Solution and Analytical Solution')
        plt.xlabel('t')
        plt.ylabel('|$y_{numeric}-y_{exact}$|')
        plt.xlim([t[0], t[-1]])
        plt.grid()
        plt.legend(loc=1)
        plt.show()