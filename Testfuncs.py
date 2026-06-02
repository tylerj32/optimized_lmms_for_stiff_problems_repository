import numpy as np
import scipy.optimize as opt
import sympy as sp
tau1 = np.arange(0.01,1,0.1) # tau = time step
def lmm1(t,tau):
    # Setting up trapezoidal
    x_num = np.zeros_like(t)
    x_num[0] = 1.0  # Initial condition for x(0) = 1
    # Leapfrog loop (euler)
    for n in range(1, len(t) - 1):
        x_num[n + 1] = x_num[n - 1] - 2 * tau * x_num[n]
    # Setting up leapfrog with Runge-Kutta
    x_num1 = np.zeros_like(t)
    x_num1[0] = 1.0
    # First step
    y1 = -x_num1[0]
    y2 = -(x_num1[0] + .5*tau*y1)
    y3 = -(x_num1[0] + .5*tau*y2)
    y4 = -(x_num1[0] + .5*tau*y3)
    x_num1[1] = x_num1[0] + tau/6*(y1 + 2*y2 + 2*y3 + y4)
    # Leapfrog loop (runge-kutta)
    for n in range(1, len(t) - 1):
        x_num1[n + 1] = x_num1[n - 1] - 2 * tau * x_num1[n]
    return x_num, x_num1
def exp_euler(x,y,vx,vy,n,k):
    x[n+1] = x[n] + k*vx[n]
    y[n+1] = y[n] + k*vy[n]
    vx[n+1] = vx[n] - k*x[n]/np.sqrt((x[n]**2+y[n]**2)**3)
    vy[n+1] = vy[n] - k*y[n]/np.sqrt((x[n]**2+y[n]**2)**3)
    return x,y,vx,vy
def implicit_residual(U,x_n,y_n,vx_n,vy_n,k):
    x1,y1,vx1,vy1 = U
    r = np.sqrt(x1**2+y1**2)+1e-10
    return [
        x1 - (x_n + k * vx1),
        y1 - (y_n + k * vy1),
        vx1 - (vx_n - k * x1/r**3),
        vy1 - (vy_n - k * y1/r**3)
    ]
def imp_euler(x,y,vx,vy,n,k):
    U0 = [x[n], y[n], vx[n], vy[n]]  # Initial guess
    sol = opt.fsolve(
        implicit_residual,
        U0,
        args=(x[n],y[n],vx[n],vy[n],k)
    )
    x[n+1],y[n+1],vx[n+1],vy[n+1] = sol
    return x,y,vx,vy
def mid_residual(U,x_n,y_n,vx_n,vy_n,k):
    x1,y1,vx1,vy1 = U
    xm = (x_n+x1)/2
    ym = (y_n+y1)/2
    vxm = (vx_n+vx1)/2
    vym = (vy_n+vy1)/2
    r = np.sqrt(xm**2+ym**2)+1e-10
    axm = -xm / r**3
    aym = -ym / r**3
    return [
        x1 - (x_n + k * vxm),
        y1 - (y_n + k * vym),
        vx1 - (vx_n + k * axm),
        vy1 - (vy_n + k * aym)
    ]
def mid_rule(x,y,vx,vy,n,k):
    U0 = [x[n], y[n], vx[n], vy[n]]  # Initial guess
    sol = opt.fsolve(
        mid_residual,
        U0,
        args=(x[n], y[n], vx[n], vy[n], k)
    )
    x[n + 1], y[n + 1], vx[n + 1], vy[n + 1] = sol
    return x,y,vx,vy
def symp_euler(x,y,vx,vy,n,k):
    x[n+1] = x[n] + k*vx[n]
    y[n+1] = y[n] + k*vy[n]
    r = np.sqrt(x[n+1] ** 2 + y[n+1] ** 2) + 1e-10
    vx[n+1] = vx[n]-k*x[n+1] / r**3
    vy[n+1] = vy[n]-k*y[n+1] / r**3
    return x,y,vx,vy
def leapfrog(x,y,vx_half,vy_half,n,k):
    x[n+1] = x[n] + k * vx_half[n]
    y[n+1] = y[n] + k * vy_half[n]
    r = np.sqrt(x[n+1] ** 2 + y[n+1] ** 2) + 1e-10
    ax = -x[n+1] / r**3
    ay = -y[n+1] / r**3
    vx_half[n+1] = vx_half[n] + k * ax
    vy_half[n+1] = vy_half[n] + k * ay
    return x,y,vx_half,vy_half
def A_stable_test(a,b,N=2000):
    theta = np.linspace(0,2*np.pi,N)
    xi = np.exp(1j*theta)
    rho = sum(a[j] * xi**j for j in range(len(a)))
    sigma = sum(b[j] * xi**j for j in range(len(b)))
    func = rho/sigma
    if np.any(func.real < -1e-6):
        print('This LMM is NOT A-stable. Min Real:', np.nanmin(func.real))
        return func, False
    else:
        print('This LMM is A-stable. Min Real:', np.nanmin(func.real))
        return func, True
def stability_region_mask(a, b, xlim=(-6, 6), ylim=(-6, 6), N=300):
    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    stable = np.zeros_like(Z, dtype=bool)
    for i in range(N):
        for j in range(N):
            z = Z[i, j]
            coeffs = a - z*b
            roots = np.roots(coeffs[::-1])
            stable[i, j] = np.all(np.abs(roots) <= 1 + 1e-8)
    return X, Y, stable
def L_stable_test(a,b):
    xi = sp.symbols('xi')
    rho = sum(a[j] * xi**j for j in range(len(a)))
    sigma = sum(b[j] * xi**j for j in range(len(b)))
    rho_poly = sp.Poly(rho,xi)
    sigma_poly = sp.Poly(sigma,xi)
    if rho_poly.degree() != sigma_poly.degree():
        return False, [rho_poly.degree(), sigma_poly.degree()]
    coeffs = [complex(c) for c in sigma_poly.all_coeffs()]
    roots = np.roots(coeffs)
    max_mod = max(abs(r) for r in roots)
    return abs(max_mod)-1 < -1e-6, roots