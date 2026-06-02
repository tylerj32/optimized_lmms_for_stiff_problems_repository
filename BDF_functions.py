import numpy as np
import scipy.optimize as opt
h1 = 0.1 # tau = time step
t1 = np.arange(0, 3+h1, h1)
lam1 = -10**6
# n-cos(t0) => 1.5-1
tan = np.arange(0, 3,.01)
x_exact = np.exp(lam1*t1)/2 + np.cos(t1) # Exact Solution
x_exact_comp = np.cos(t1) # Exact Solution
x_an = np.exp(lam1*tan)/2 + np.cos(tan)
x_an_comp = np.cos(tan)
def lmm1(t,h,lam):
    # Setting up trapezoidal
    x_num = np.zeros_like(t)
    x_num[0] = 1.5  # Initial condition for x(0) = 1
    for n in range(len(t) - 1):
        x_num[n + 1] = (x_num[n]*(1+(h*lam)/2)
                        - h/2*(lam*(np.cos(t[n])+np.cos(t[n+1]))
                                    +np.sin(t[n])+np.sin(t[n+1])))/(1-(h*lam)/2)
    # Setting up implicit euler
    x_num1 = np.zeros_like(t)
    x_num1[0] = 1.5
    for n in range(len(t) - 1):
        x_num1[n + 1] = (x_num1[n] - h*(lam*np.cos(t[n+1])+np.sin(t[n+1])))/(1-h*lam)
    return x_num, x_num1
# Testing

def LMM_solver(t,h,alpha,beta,t0,f):
    lmm_y = np.zeros_like(t)
    s = len(alpha) - 1
    lmm_y[0] = t0 # Initial condition
    # First step(s) using TR-BDF2
    for i in range(1,s):
        u_n = lmm_y[i-1]
        t_n = t[i-1]
        t_np1 = t[i]
        # Stage 1: TR half-step
        def make_tr(u_n_cap, h_cap, t_cap):
            def fn(u_star):
                return u_star - (u_n_cap + h_cap / 4 * (f(t_cap, u_n_cap) + f(t_cap + h_cap / 2, u_star)))

            return fn
        tr = make_tr(u_n, h, t_n)
            # h/4*(f(t_n, u_n)) + f(t_n+h/2, u_star)) -> bad code
        u_star = opt.fsolve(tr, u_n)[0]
        # Stage 2: BDF2 full step
        def make_trbdf2(u_star_cap, u_n_cap, h_cap, t_cap):
            def fn(u_np1):
                return u_np1 - 1 / 3 * (4 * u_star_cap - u_n_cap + h_cap * (f(t_cap, u_np1)))
            return fn
        trbdf2 = make_trbdf2(u_star,u_n,h,t_np1)
        lmm_y[i] = opt.fsolve(trbdf2, u_star)[0]
    # Adding the rest of the steps
    for n in range(len(t) - s):
        def lmm(us):
            full_lmm = 0.0
            # alpha terms
            for j in range(s):
                full_lmm += alpha[j] * lmm_y[n + j]
            full_lmm += alpha[s] * us
            # beta terms
            for j in range(s):
                full_lmm -= h * beta[j] * f(t[n+j],lmm_y[n+j])
            full_lmm -= h * beta[s] * f(t[n+s],us)
            return full_lmm
        # Initial guess for fsolve: previous step
        lmm_y[n + s] = opt.fsolve(lmm, lmm_y[n + s - 1])[0]
    return lmm_y
def bdf2_og(t,h,lam):
    bdf_y = np.zeros_like(t)
    bdf_y[0] = 1.5 # Initial condition
    # First step: Implicit Euler
    bdf_y[1] = (bdf_y[0] - h*(lam*np.cos(t[1])+np.sin(t[1])))/(1-h*lam)
    # Filling in rest of steps
    for n in range(len(bdf_y) - 2):
        bdf_y[n+2] = (4/3*bdf_y[n+1] - 1/3*bdf_y[n] - (2*h)/3*(lam*np.cos(t[n+2]) + np.sin(t[n+2])))/(1 - (2*h*lam)/3)
    return bdf_y
def bdf3_like(t,h,lam):
    bdf_y = np.zeros_like(t)
    bdf_y[0] = 1.5  # Initial condition
    # First two steps: Implicit Euler
    for i in range(1,3):
        bdf_y[i] = (bdf_y[i-1] - h * (lam * np.cos(t[i]) + np.sin(t[i]))) / (1 - h * lam)
    # Filling in rest of steps
    for n in range(len(bdf_y) - 3):
        bdf_y[n + 3] = (3/2*bdf_y[n+2]-3/5*bdf_y[n+1]+1/10*bdf_y[n]-(3*h)/5
                        *(lam*np.cos(t[n+3])+np.sin(t[n+3]))) / (1 - (3 * h * lam) / 5)
    return bdf_y
def LMM_solver2(t,alpha,beta,t0,f,jac):
    h = np.diff(t)
    m = len(t0)  # dimension 3
    lmm_y = np.zeros((len(t), m))
    lmm_y[0] = np.array(t0) # Initial Condition
    s = len(alpha(h, len(h)-1)) - 1 if callable(alpha) else len(alpha) - 1
    # Jacobian functions for TR-BDF2
    def make_stage_jacobian(scale, t):
        def J(u):
            return np.eye(m) - scale * jac(t, u)
        return J
    # First step(s) using TR-BDF2
    for i in range(1,s):
        u_n = lmm_y[i-1]
        t_np1 = t[i]
        t_n = t[i-1]
        h_n = t[i] - t[i - 1]
        # Stage 1: TR half-step
        def make_tr(u_n_cap, h_cap, t_cap):
            def fn(u_star):
                return u_star - (u_n_cap + h_cap / 4 * (f(t_cap, u_n_cap) + f(t_cap + h_cap / 2, u_star)))
            return fn
        jac_tr = make_stage_jacobian(h_n / 4, t_n + h_n / 2)
        tr = make_tr(u_n, h_n, t_n)
        u_star, info, ier, mesg = opt.fsolve(tr, u_n, fprime=jac_tr, full_output=True)
        if ier != 1:
            print(f"TR stage 1 failed at startup step {i}: {mesg}")
        # Stage 2: BDF2 full step
        jac_bdf2 = make_stage_jacobian(h_n / 3, t_np1)
        def make_trbdf2(u_star_cap, u_n_cap, h_cap, t_cap):
            def fn(u_np1):
                return u_np1 - 1 / 3 * (4 * u_star_cap - u_n_cap + h_cap * (f(t_cap, u_np1)))
            return fn
        trbdf2 = make_trbdf2(u_star,u_n,h_n,t_np1)
        u_np1, info, ier, mesg = opt.fsolve(trbdf2, u_star, fprime=jac_bdf2, full_output=True)
        if ier != 1:
            print(f"TR-BDF2 stage 2 failed at startup step {i}: {mesg}")
        lmm_y[i] = u_np1
    # Adding the rest of the steps
    for n in range(len(t) - s):
        h_n = t[n + s] - t[n + s - 1]
        alpha_n = alpha(h, n + s - 1) if callable(alpha) else alpha
        beta_n = beta(h, n + s - 1) if callable(beta) else beta

        def lmm(u_new, _alpha=alpha_n, _beta=beta_n):
            full_lmm = np.zeros_like(u_new)
            for j in range(s):
                full_lmm += _alpha[j] * lmm_y[n + j]
            full_lmm += _alpha[s] * u_new
            for j in range(s):
                full_lmm -= h_n * _beta[j] * f(t[n + j], lmm_y[n + j])
            full_lmm -= h_n * _beta[s] * f(t[n + s], u_new)
            return full_lmm

        jac_lmm = lambda u, _a=alpha_n, _b=beta_n: \
            _a[s] * np.eye(m) - h_n * _b[s] * jac(t[n + s], u)

        # Extrapolated guess
        if n > 0:
            guess = lmm_y[n + s - 1] + h_n / h[n + s - 2] * (lmm_y[n + s - 1] - lmm_y[n + s - 2])
        else:
            guess = lmm_y[n + s - 1]

        result, info, ier, mesg = opt.fsolve(lmm, guess, fprime=jac_lmm, full_output=True)
        residual = np.linalg.norm(info['fvec'])
        if ier != 1 and residual > 1e-10:
            print(f"fsolve failed at t={t[n + s]:.4f}: ier={ier}, |fvec|={residual:.2e}, msg={mesg}")
        print(f'Finished loop {n + 1} of {len(t) - s}')
        lmm_y[n + s] = result
    return lmm_y