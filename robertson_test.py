import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import BDF_functions as bf
from scipy.integrate import solve_ivp
# In this script, we will attempt to solve the Robertson problem using the BDF-like methods
# The Robertson problem is defined below:
# dy_1/dt = -0.04*y_1 + 10^4*y_2*y_3
# dy_2/dt = 0.04*y1 - 10^4*y_2*y_3 - 3*10^7*y_2^2
# dy_3/dt = 3*10^7*y_2^2
# We also have the following initial conditions: y_1(0) = 1, y_2(0) = 0, y_3(0) = 0
# eps=0.01
t_1 = np.logspace(-6, 8, 2000)
# t1 = 1+eps*np.random.random_sample(t_1.shape)
# t_1 *=t1
t0 = [1.0, 0.0, 0.0]
def f(t, u):
    y1, y2, y3 = u
    return np.array([
        -0.04*y1 + 1e4*y2*y3,
        0.04*y1 - 1e4*y2*y3 - 3e7*y2**2,
        3e7*y2**2
    ])
def jac(t, u):
    y1, y2, y3 = u
    return np.array([
        [-0.04,        1e4*y3,     1e4*y2],
        [0.04, -1e4*y3 - 6e7*y2,  -1e4*y2],
        [0.0,         6e7*y2,      0.0]
    ])
def alpha_bdf2(h, n):
    omega_n = h[n] / h[n-1]
    return np.array([(omega_n**2)/(1+2*omega_n), -((1+omega_n)**2)/(1+2*omega_n), 1])

def beta_bdf2(h, n):
    omega_n = h[n] / h[n - 1]
    return np.array([0, 0, (1+omega_n)/(1+2*omega_n)])

def alpha_bdf3(h, n):
    omega_n = h[n] / h[n-1]
    omega_n1 = h[n-1]/h[n-2]
    a0 = -((omega_n1**2* (1 + omega_n1) *omega_n**3)/((1 + omega_n) *(1 + (1 + omega_n1)* omega_n)))
    a1 = omega_n1**2/(1 + omega_n1) + omega_n1**2 *omega_n
    a2 = -1 - omega_n1 - (omega_n1 *(1 + omega_n1)* omega_n)/(1 + omega_n)
    a3 = 1 + omega_n1/(1 + omega_n1) + (omega_n1 *omega_n)/(1 + (1 + omega_n1) *omega_n)
    return np.array([a0,a1,a2,a3])
def beta_bdf3(h, n):
    return np.array([0,0,0,1])
beta_3 = 3/5
def alpha_bdfl3(h, n):
    omega_n = h[n] / h[n-1]
    a0 = -((omega_n**2 *(-beta_3 + (1 - 2 *beta_3)* omega_n + omega_n**2))/(1 + omega_n))
    a1 = -beta_3 - (-1 + beta_3)* omega_n + (1 - 2 *beta_3)* omega_n**2 + omega_n**3
    a2 = 1/(1 + omega_n)* (-1 + beta_3 + 2 *(-1 + beta_3) *omega_n + 2* (-1 + beta_3) *omega_n**2 - omega_n**3)
    return np.array([a0,a1,a2,1])

def beta_bdfl3(h, n):
    return np.array([0,0,0,beta_3])

def alpha_bdfl3_bdfopt(h, n):
    n = n if n >= 0 else len(h) + n  # normalize negative index
    omega_n = h[n] / h[n-1]
    omega_n1 = h[n+1]/h[n] if n+1<len(h) else omega_n
    a0, a1, a2, a3 = [-1 / 2 * (omega_n ** 3 * omega_n1 * (1 + omega_n1)) /
                      ((1 + omega_n) * (1 + omega_n * (1 + omega_n1))),
                      (omega_n ** 2 / (1 + omega_n) + omega_n1 *
                       (omega_n + (1 + omega_n1) ** (-1))) / 2,
                      (-1 - omega_n - ((1 + omega_n1) *
                        (1 + omega_n * (1 + omega_n1))) / ((1 + omega_n) *
                            omega_n1)) / 2, ((1 + 2 * omega_n) / (1 + omega_n) +
                                (1 + 2 * omega_n1 + omega_n * (1 + 4 * omega_n1 +
                                    3 * omega_n1 ** 2)) / (omega_n1 * (1 + omega_n1) *
                                        (1 + omega_n * (1 + omega_n1)))) / 2]
    return np.array([a0,a1,a2,a3])

def beta_bdfl3_bdfopt(h, n):
    return np.array([0,0,0,1])


def alpha_bdfl4(h, n):
    omega_n = h[n] / h[n-1]
    omega_n1 = h[n+1]/h[n] if n+1<len(h) else omega_n
    omega_n2 = h[n + 2] / h[n + 1] if n + 2 < len(h) else omega_n1
    a0 = (((1 - 1/np.sqrt(2)) *omega_n**4 *omega_n1**3* omega_n2**2* (1 + omega_n2)* (1 + omega_n1* (1 + omega_n2)))
          /((1 + omega_n)* (1 + omega_n* (1 + omega_n1))* (1 + omega_n* (1 + omega_n1* (1 + omega_n2)))))
    a1 = -(((-(5/2) + 2*np.sqrt(2)) *omega_n**3* omega_n1**2* (1 + omega_n1))/((1 + omega_n)* (1 + omega_n
            * (1 + omega_n1)))) + ((-2 + np.sqrt(2))* omega_n1**3* omega_n2**2* (1 + omega_n2) *(1 + omega_n
                    * (1 + omega_n1* (1 + omega_n2))))/(2* (1 + omega_n1)* (1 + omega_n1* (1 + omega_n2)))
    a2 = ((5/2 - 3/np.sqrt(2))* omega_n**2)/(1 + omega_n) + (-(5/2) + 2 *np.sqrt(2))* omega_n1**2 *(omega_n + 1/(1
            + omega_n1)) - ((-2 + np.sqrt(2))* omega_n2**2* (1 + omega_n1* (1 + omega_n2)) *(1 + omega_n* (1 + omega_n1
                    * (1 + omega_n2))))/(2 *(1 + omega_n)* (1 + omega_n2))
    a3 = 1/2 *((-5 + 3 *np.sqrt(2))* (1 + omega_n) - ((-5 + 4 *np.sqrt(2))* (1 + omega_n1)* (1 + omega_n* (1
            + omega_n1)))/(1 + omega_n) + ((-2 + np.sqrt(2))* (1 + omega_n2) *(1 + omega_n1* (1 + omega_n2)) *(1
                    + omega_n* (1 + omega_n1 *(1 + omega_n2))))/((1 + omega_n1)* (1 + omega_n* (1 + omega_n1))))
    a4 = ((5/2 - 3/np.sqrt(2)) *(1 + 2 *omega_n))/(1 + omega_n) + (-(5/2) + 2 *np.sqrt(2))* (1 + omega_n1* (1/(1
            + omega_n1) + omega_n/(1 + omega_n *(1 + omega_n1)))) + (1 - 1/np.sqrt(2))* (1 + omega_n2* (1/(1 + omega_n2)
                    + omega_n1* (1/(1 + omega_n1* (1 + omega_n2)) + omega_n/(1 + omega_n* (1 + omega_n1 *(1
                            + omega_n2))))))
    return np.array([a0,a1,a2,a3,a4])

def beta_bdfl4(h, n):
    return np.array([0,0,0,0,1])

sqrt5 = np.sqrt(5)
def alpha_bdfl5(h, n):
    omega_n = h[n] / h[n-1]
    omega_n1 = h[n+1]/h[n] if n+1<len(h) else omega_n
    omega_n2 = h[n + 2] / h[n + 1] if n + 2 < len(h) else omega_n1
    omega_n3 = h[n + 3] / h[n + 2] if n + 3 < len(h) else omega_n2
    print('omega_n =', omega_n) if n == 1995 else None
    a0, a1, a2, a3, a4, a5 = [
        ((-3 + np.sqrt(5)) * omega_n ** 5 * omega_n1 ** 4 * omega_n2 ** 3 * omega_n3 ** 2 * (1 + omega_n3) * (
                    1 + omega_n2 * (1 + omega_n3)) *
         (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3)))) / (
                    4 * (1 + omega_n) * (1 + omega_n * (1 + omega_n1)) *
                    (1 + omega_n * (1 + omega_n1 * (1 + omega_n2))) * (
                                1 + omega_n * (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))))),
        (omega_n1 ** 3 * omega_n2 ** 2 * (((-35 + 17 * np.sqrt(5)) * omega_n ** 4 * (1 + omega_n2) * (
                    1 + omega_n1 * (1 + omega_n2)) ** 2) /
                                              ((1 + omega_n) * (1 + omega_n * (1 + omega_n1)) * (
                                                        1 + omega_n * (1 + omega_n1 * (1 + omega_n2)))) -
                                              (5 * (-3 + np.sqrt(5)) * omega_n1 * omega_n2 * omega_n3 ** 2 * (
                                                        1 + omega_n3) * (1 + omega_n2 * (1 + omega_n3)) *
                                               (1 + omega_n * (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))))) /
                                              ((1 + omega_n1) * (
                                                        1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3)))))) / (
                    20 * (1 + omega_n1 * (1 + omega_n2))),
        ((2 * (-10 + 3 * np.sqrt(5)) * omega_n ** 3 * omega_n1 ** 2 * (1 + omega_n1)) / (
                    (1 + omega_n) * (1 + omega_n * (1 + omega_n1))) -
         ((-35 + 17 * np.sqrt(5)) * omega_n1 ** 3 * omega_n2 ** 2 * (1 + omega_n2) * (
                     1 + omega_n * (1 + omega_n1 * (1 + omega_n2)))) /
         ((1 + omega_n1) * (1 + omega_n1 * (1 + omega_n2))) + (
                     5 * (-3 + np.sqrt(5)) * omega_n2 ** 3 * omega_n3 ** 2 * (1 + omega_n3) *
                     (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))) * (
                                 1 + omega_n * (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))))) /
         ((1 + omega_n) * (1 + omega_n2) * (1 + omega_n2 * (1 + omega_n3)))) / 20,
        ((-2 * (-10 + 3 * np.sqrt(5)) * omega_n ** 2) / (1 + omega_n) + 2 * (10 - 3 * np.sqrt(5)) * omega_n1 ** 2 * (
                    omega_n + (1 + omega_n1) ** (-1)) +
         ((-35 + 17 * np.sqrt(5)) * omega_n2 ** 2 * (1 + omega_n1 * (1 + omega_n2)) * (
                     1 + omega_n * (1 + omega_n1 * (1 + omega_n2)))) /
         ((1 + omega_n) * (1 + omega_n2)) - (
                     5 * (-3 + np.sqrt(5)) * omega_n3 ** 2 * (1 + omega_n2 * (1 + omega_n3)) *
                     (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))) * (
                                 1 + omega_n * (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))))) /
         ((1 + omega_n1) * (1 + omega_n * (1 + omega_n1)) * (1 + omega_n3))) / 20,
        (2 * (-10 + 3 * np.sqrt(5)) * (1 + omega_n) + (
                    2 * (-10 + 3 * np.sqrt(5)) * (1 + omega_n1) * (1 + omega_n * (1 + omega_n1))) / (1 + omega_n) -
         ((-35 + 17 * np.sqrt(5)) * (1 + omega_n2) * (1 + omega_n1 * (1 + omega_n2)) * (
                     1 + omega_n * (1 + omega_n1 * (1 + omega_n2)))) /
         ((1 + omega_n1) * (1 + omega_n * (1 + omega_n1))) + (
                     5 * (-3 + np.sqrt(5)) * (1 + omega_n3) * (1 + omega_n2 * (1 + omega_n3)) *
                     (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))) * (
                                 1 + omega_n * (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))))) /
         ((1 + omega_n2) * (1 + omega_n1 * (1 + omega_n2)) * (
                     1 + omega_n * (1 + omega_n1 * (1 + omega_n2))))) / 20,
        ((-2 * (-10 + 3 * np.sqrt(5)) * (1 + 2 * omega_n)) / (1 + omega_n) +
         2 * (10 - 3 * np.sqrt(5)) * (
                     1 + omega_n1 * ((1 + omega_n1) ** (-1) + omega_n / (1 + omega_n * (1 + omega_n1)))) +
         (-35 + 17 * np.sqrt(5)) * (1 + omega_n2 * (
                            (1 + omega_n2) ** (-1) + omega_n1 * ((1 + omega_n1 * (1 + omega_n2)) ** (-1) +
                                                                    omega_n / (1 + omega_n * (
                                        1 + omega_n1 * (1 + omega_n2)))))) + 5 * (3 - np.sqrt(5)) * omega_n1 * (
             omega_n2) * omega_n3 *
         ((1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))) ** (-1) + (
                     1 + 2 * omega_n3 + omega_n2 * (1 + 4 * omega_n3 + 3 * omega_n3 ** 2) +
                     omega_n * (1 + 2 * omega_n3 + omega_n2 * (1 + 4 * omega_n3 + 3 * omega_n3 ** 2) +
                                  omega_n1 * (1 + 2 * omega_n3 + omega_n2 ** 2 * (1 + omega_n3) ** 2 * (
                                 1 + 4 * omega_n3) + omega_n2 * (2 + 8 * omega_n3 + 6 * omega_n3 ** 2)))) /
          (omega_n1 * omega_n2 * omega_n3 * (1 + omega_n3) * (1 + omega_n2 * (1 + omega_n3)) *
           (1 + omega_n * (1 + omega_n1 * (1 + omega_n2 * (1 + omega_n3))))))) / 20
    ]
    return np.array([a0,a1,a2,a3,a4,a5])

def beta_bdfl5(h, n):
    return np.array([0,0,0,0,0,1])
if __name__ == "__main__":
    bdf2 = bf.LMM_solver2(t_1, alpha_bdf2, beta_bdf2, t0, f, jac)
    bdf3 = bf.LMM_solver2(t_1, alpha_bdf3, beta_bdf3, t0, f, jac)
    bdfl3 = bf.LMM_solver2(t_1, alpha_bdfl3_bdfopt, beta_bdfl3_bdfopt, t0, f, jac)
    # bdfl3err = bf.LMM_solver2(t_1,alpha_bdfl3_errcons,beta_bdfl3_errcons,t0, f, jac)
    bdfl4 = bf.LMM_solver2(t_1,alpha_bdfl4,beta_bdfl4,t0, f, jac)
    bdfl5 = bf.LMM_solver2(t_1,alpha_bdfl5,beta_bdfl5,t0, f, jac)
    # True solution
    ref = solve_ivp(f, [t_1[0], t_1[-1]], t0, method='Radau', jac=jac, rtol=2.220446049250313e-14, atol=1e-20, dense_output=True)
    ref_vals = ref.sol(t_1[:-1])  # interpolated solution at t_1
    print('Max value of y2 for BDFL5: ',np.max(bdfl5[:,1]))
    sol = np.array([bdf2,bdf3,bdfl3,bdfl4,bdfl5])
    t_1 = t_1[:-1]
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    method_handles = []
    y1_exact = ref_vals[0]
    y2_exact = ref_vals[1]
    y3_exact = ref_vals[2]
    for i in range(len(sol)):
        y1 = sol[i,:-1,0]
        y2 = sol[i,:-1,1]
        y3 = sol[i,:-1,2]
        if i<=1:
            label = f'BDF{i + 2}'
        else:
            label = f'BDFL${i+1}_2$'
        method_handles.append(
            Line2D([0], [0], color=default_colors[i], lw=2, label=label)
        )
        plt.semilogx(t_1, y1, color=default_colors[i], label=label, linestyle='-')
        plt.semilogx(t_1, y2*1e4, color=default_colors[i], linestyle='--')
        plt.semilogx(t_1, y3, color=default_colors[i], linestyle=':')
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
    fig, axs = plt.subplots(3, 1, sharex=True)  # 3 rows, 1 column
    line = ['-','--',':','-.',(0, (3, 1,1,1)),(0,(1,5)),(0,(1,1)),(5,(10,3)),(0,(5,10)),(0,(5,5))]
    for i in range(len(sol)):
        y1 = sol[i,:-1,0]
        y2 = sol[i,:-1,1]
        y3 = sol[i,:-1,2]
        # label = f'BDF{i+2}' if i <= 1 else f'BDFL{i+1}'
        if i<=1:
            label = f'BDF{i + 2}'
        else:
            label = f'BDFL${i+1}_2$'
        axs[0].loglog(t_1, np.abs(y1-y1_exact), color=default_colors[i],linestyle=line[i], label=label)
        axs[1].loglog(t_1, np.abs(y2-y2_exact), color=default_colors[i], linestyle=line[i])
        axs[2].loglog(t_1, np.abs(y3-y3_exact), color=default_colors[i], linestyle=line[i])
    # Labels for each subplot
    axs[0].set_ylabel(r'$|y_1 - y_{1\text{(Exact)}}|$')
    axs[1].set_ylabel(r'$|y_2 - y_{2\text{(Exact)}}|$')
    axs[2].set_ylabel(r'$|y_3 - y_{3\text{(Exact)}}|$')
    axs[2].set_xlabel('t')

    # axs[0].set_ylim(1e-13,1e-4)

    # Only one legend (usually top plot)
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].grid()
    axs[1].grid()
    plt.tight_layout()
    plt.grid()
    plt.show()