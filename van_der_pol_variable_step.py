import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# from scipy.signal import hilbert
import scipy.optimize as opt
import BDF_like_comparison as blc
from scipy.integrate import solve_ivp
import robertson_test as rt
# In this script, we will attempt to solve a system of ODEs defined below:
# dy_1/dt = y2
# dy_2/dt = \mu*(1-y_1^2)y2-y1
# This denotes a self oscillating system that dissipates. It becomes very stiff as \mu = 500
# We will also use a dynamic system of variable step size LMMs to better solve this problem
y0 = [2.0, 0.0]
# Initial step size
h1 = .001
# Step size limits (for now)
h_min = 1e-5
h_max = 100.0
omega_max = 1.5
omega_min = 0.1
# 2000
t_final = 1000
# Tolerance
tol = 1e-6
# Defining the differential equation (and its Jacobian)
mu = 1000
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
# Adding the specialized solver for this method
def LMM_step_solver(t,h,alpha,beta,lmm_y,f,jac):
    m = 2  # dimension of t0: 2
    # Determine s safely using a dummy h array with enough elements
    _h_dummy = [h[-1]] * 3
    s = len(alpha(_h_dummy, -1)) - 1
    h_n = h[-1]
    t_new = t[-1] + h_n
    # Evaluating Implicit Euler

    def ie_residual(y):
        return y - lmm_y[-1] - h_n * f(t_new, y)

    jac_ie = lambda y: np.eye(m) - h_n * jac(t_new, y)
    y_new = opt.fsolve(ie_residual, lmm_y[-1], fprime=jac_ie)
    # For the first k-1 steps we use TR-BDF2
    if len(t)<s:
        # Jacobian functions for TR-BDF2
        def make_stage_jacobian(scale, t):
            def J(u):
                return np.eye(m) - scale * jac(t, u)
            return J
        # First step(s) using TR-BDF2
        u_n = lmm_y[-1]
        h_n = h[-1]
        t_n = t[-1]
        t_nm1 = t[-2] if len(t)>1 else t[0]
        # Stage 1: TR half-step
        def make_tr(u_n_cap, h_cap, t_cap):
            def fn(u_star):
                return u_star - (u_n_cap + h_cap / 4 * (f(t_cap, u_n_cap) + f(t_cap + h_cap / 2, u_star)))
            return fn
        jac_tr = make_stage_jacobian(h_n / 4, t_nm1 + h_n / 2)
        tr = make_tr(u_n, h_n, t_nm1)
        u_star, info, ier, mesg = opt.fsolve(tr, u_n, fprime=jac_tr, full_output=True)
        if ier != 1:
            print(f"TR stage 1 failed at startup step {i}: {mesg}")
        # Stage 2: BDF2 full step
        jac_bdf2 = make_stage_jacobian(h_n / 3, t_n)
        def make_trbdf2(u_star_cap, u_n_cap, h_cap, t_cap):
            def fn(u_np1):
                return u_np1 - 1 / 3 * (4 * u_star_cap - u_n_cap + h_cap * (f(t_cap, u_np1)))
            return fn
        trbdf2 = make_trbdf2(u_star,u_n,h_n,t_n)
        # Safety check for convergence in fsolve
        result, info, ier, mesg = opt.fsolve(trbdf2, u_star, fprime=jac_bdf2, full_output=True)
        if ier != 1:
            print(f"TR-BDF2 stage 2 failed at startup step {i}: {mesg}")
        # Embedded error estimate
        r_n = np.linalg.norm(result - y_new)
        r_n = max(r_n, 1e-14) # The max prevents division by zero
    # Once s<len(t) then we start adding the rest of the steps via the LMM
    else:

        alpha_n = alpha(h, -1)
        beta_n = beta(h, -1)

        def lmm(u_new, _alpha=alpha_n, _beta=beta_n):
            full_lmm = np.zeros_like(u_new)
            for j in range(s):
                full_lmm += _alpha[j] * lmm_y[-s+j]
            full_lmm += _alpha[s] * u_new
            for j in range(s):
                full_lmm -= h_n * _beta[j] * f(t[-s+j], lmm_y[-s+j])
            full_lmm -= h_n * _beta[s] * f(t[-1], u_new)
            return full_lmm

        jac_lmm = lambda u, _a=alpha_n, _b=beta_n: \
            _a[s] * np.eye(m) - h_n * _b[s] * jac(t[-1], u)

        # Extrapolated guess
        guess = lmm_y[-1] + h_n / h[-2] * (lmm_y[-1] - lmm_y[-2])
        result, info, ier, mesg = opt.fsolve(lmm, guess, fprime=jac_lmm, full_output=True)
        residual = np.linalg.norm(info['fvec'])
        if ier != 1 and residual > 1e-10:
            print(f"fsolve failed at t={t[-1]:.4f}: ier={ier}, |fvec|={residual:.2e}, msg={mesg}")
        # Scale by (1 + omega) to account for variable
        omega_n = h_n / h[-2]
        r_n = np.linalg.norm(result-y_new)/(1+omega_n)
        r_n = max(r_n, 1e-14) # The max prevents division by zero
    h_new = (tol / r_n) ** (1 / 2) * h_n
    # Controlling the size of the steps
    h_new = min(h_new, omega_max * h_n)
    h_new = max(h_new, omega_min * h_n)
    h_new = max(h_min, min(h_max, h_new))
    return result,float(h_new)
def full_lmm_solver(alpha,beta,f,jac):
    lmm_y = np.array([y0], dtype=float)
    t = [0.0] # t starts at 0s and 0+h_n seconds
    h=[h1] # BDFL3 and 4 need more than one h_n
    quarter_printed = False
    half_printed = False
    threequarter_printed = False
    ninety_printed = False
    while t[-1]<=t_final:
        lmm_y_new,h_new = LMM_step_solver(t,h,alpha,beta,lmm_y,f,jac)
        lmm_y = np.append(lmm_y, [lmm_y_new], axis=0)
        h.append(h_new)
        t.append(t[-1]+h_new)
        current_t = t[-1]
        if (not quarter_printed) and current_t >= 0.25 * t_final:
            print("The solver is 25% complete...")
            quarter_printed = True
        if (not half_printed) and current_t >= 0.50 * t_final:
            print("The solver is 50% complete...")
            half_printed = True
        if (not threequarter_printed) and current_t >= 0.75 * t_final:
            print("The solver is 75% complete...")
            threequarter_printed = True
        if (not ninety_printed) and current_t >= 0.9 * t_final:
            print("The solver is 90% complete...")
            ninety_printed = True
    print('Finished! :)')
    return h,t,lmm_y
# BDFL methods
print('Starting BDF2')
h,t_2,bdf2 = full_lmm_solver(rt.alpha_bdf2,rt.beta_bdf2,f,jac)
print('Starting BDFL3_2')
h2,t_3,bdflike3 = full_lmm_solver(rt.alpha_bdfl3_bdfopt,rt.beta_bdfl3_bdfopt,f,jac)
# print('Starting BDFL4')
# bdflike4 = full_lmm_solver(rt.alpha_bdfl4,rt.beta_bdfl4,f,jac)
# print('Starting BDFL5')
# bdflike5 = full_lmm_solver(rt.alpha_bdfl5,rt.beta_bdfl5,f,jac)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
sol = [bdf2,bdflike3]
t=[t_2,t_3]
ref = solve_ivp(f, [t_2[0], t_2[-1]], y0, method='Radau', jac=jac, rtol=1e-10, atol=1e-12, dense_output=True)
ref_vals = [ref.sol(t_2),ref.sol(t_3)] # interpolated solution at t_2 and t_3
h_set = [h,h2]
line1 = blc.line
method_handles = []
plt.plot(t_2,ref_vals[0][0], color='black', lw=2, label='Exact solution')
method_handles.append(
        Line2D([0], [0], color='black', lw=2, label='Exact solution')
    )
for i in range(len(sol)):
    y_1 = sol[i][:,0]
    label = 'BDF2' if i==0 else f'BDFL${i+2}_2$'
    method_handles.append(
        Line2D([0], [0], color=default_colors[i+1], lw=2, label=label,linestyle=line1[i])
    )
    plt.plot(t[i], y_1, color=default_colors[i+1], label=label, linestyle=line1[i])
legend1 = plt.legend(handles=method_handles, title="Method",loc='lower left')
plt.xlabel('t')
plt.ylabel(r'$y_2$',rotation=0)
plt.title('Numerical Result - van der Pol')
plt.tight_layout()
plt.grid()
plt.show()
# Error plot
# phase_diff_set = []
# for i, s in enumerate(sol):
#     # Reference solution evaluated on THIS method's grid
#     ref_vals_i = ref.sol(t[i])
#     y2_exact = ref_vals_i[1]
#     analytic_exact = hilbert(y2_exact)
#     phase_exact = np.unwrap(np.angle(analytic_exact))
#     # Numerical solution
#     y2_num = s[:,1]
#     analytic_sol = hilbert(y2_num)
#     phase_sol = np.unwrap(np.angle(analytic_sol))
#     phase_diff_sol = np.abs(phase_sol - phase_exact)
#     phase_diff_set.append(phase_diff_sol)
for i in range(len(sol)):
    y_1 = sol[i][:, 0]
    label = 'BDF2' if i == 0 else f'BDFL${i+2}_2$'
    plt.semilogy(t[i], np.abs(y_1-ref_vals[i][0]), color=default_colors[i+1],linestyle = line1[i],label = label)
plt.ylabel(r'$|\text{Error}|$')
plt.xlabel('t')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# Step size plot
for i in range(len(h_set)):
    label = 'BDF2' if i == 0 else f'BDFL${i+2}_2$'
    plt.semilogy(t[i], h_set[i], color=default_colors[i+1],linestyle = line1[i],label = label)
plt.ylabel(r'$h_n$',rotation=0)
plt.xlabel(r'$t$')
plt.title(r'Step size at $t_n$')
plt.legend()
plt.grid()
plt.show()
# Step size ratio plot
for i in range(len(h_set)):
    h_i = h_set[i]
    omega = np.array(h_i[1:]) / np.array(h_i[:-1])
    label = 'BDF2' if i == 0 else f'BDFL${i+2}_2$'
    plt.semilogy(t[i][1:], omega, color=default_colors[i+1],linestyle = line1[i],label = label)
plt.ylabel(r'$\omega_n$',rotation=0)
plt.xlabel(r'$t$')
plt.title(r'Step size ratio ($\omega_n$) at $t_n$')
plt.legend()
plt.grid()
plt.show()