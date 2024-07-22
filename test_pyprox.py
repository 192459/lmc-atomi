import numpy as np
from pylops import FirstDerivative
from pyproximal import L1, L2
from pyproximal.optimization.primal import LinearizedADMM

np.random.seed(1)

# Create noisy data
nx = 101
x = np.zeros(nx)
x[:nx//2] = 10
x[nx//2:3*nx//4] = -5
n = np.random.normal(0, 2, nx)
y = x + n

# Define functionals
l2 = L2(b=y)
l1 = L1(sigma=5.)
Dop = FirstDerivative(nx, edge=True, kind='backward')

# Solve functional with L-ADMM
L = np.real((Dop.H * Dop).eigs(neigs=1, which='LM')[0])
tau = 1.
mu = 0.99 * tau / L
xladmm, _ = LinearizedADMM(l2, l1, Dop, tau=tau, mu=mu,
                           x0=np.zeros_like(x), niter=200)