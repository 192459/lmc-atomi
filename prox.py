# Copyright 2023 by Tim Tsz-Kit Lau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import itertools
import random
import time

import numpy as np
from numpy.random import default_rng
import scipy
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr

import pylops
from pylops import MatrixMult, Identity
from pylops.optimization.basic import lsqr
from pylops.utils.backend import get_array_module, get_module_name

import pyproximal
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class L2_moreau_env(ProxOperator):
    r"""L2 Norm proximal operator.

    The Proximal operator of the :math:`\ell_2` norm minus 
    :math:`\lambda` times the Moreau envelope of the total variation 
    is defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2 - \lambda g_\gamma(\mathbf{x})`
    and :math:`f_\alpha(\mathbf{x}) = f(\mathbf{x}) +
    \alpha \mathbf{q}^T\mathbf{x}`.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    q : :obj:`numpy.ndarray`, optional
        Dot vector
    sigma : :obj:`float`, optional
        Multiplicative coefficient of L2 norm
    alpha : :obj:`float`, optional
        Multiplicative coefficient of dot product
    lamda : :obj:`float`, optional
        Multiplicative coefficient of Moreau envelope
    gamma : :obj:`float`, optional
        Smoothing parameter of Moreau envelope
    qgrad : :obj:`bool`, optional
        Add q term to gradient (``True``) or not (``False``)
    niter : :obj:`int` or :obj:`func`, optional
        Number of iterations of iterative scheme used to compute the proximal.
        This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has
        been invoked before and returns the ``niter`` to be used.
    rtol : :obj:`float`, optional
        Relative tolerance for stopping criterion.
    x0 : :obj:`np.ndarray`, optional
        Initial vector
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.
    densesolver : :obj:`str`, optional
        Use ``numpy``, ``scipy``, or ``factorize`` when dealing with explicit
        operators. The former two rely on dense solvers from either library,
        whilst the last computes a factorization of the matrix to invert and
        avoids to do so unless the :math:`\tau` or :math:`\sigma` paramets
        have changed. Choose ``densesolver=None`` when using PyLops versions
        earlier than v1.18.1 or v2.0.0
    **kwargs_solver : :obj:`dict`, optional
        Dictionary containing extra arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver when using
        numpy data (or :py:func:`pylops.optimization.solver.lsqr` and
        when using cupy data)

    Notes
    -----
    The L2 - Moreau envelope proximal operator is defined as:

    .. math::

        prox_{\tau f_\alpha}(\mathbf{x}) =
        \left(\mathbf{I} + \tau \sigma \mathbf{Op}^T \mathbf{Op} \right)^{-1}
        \left( \mathbf{x} + \tau \sigma \mathbf{Op}^T \mathbf{b} -
        \tau \alpha \mathbf{q}\right)

    when both ``Op`` and ``b`` are provided. This formula shows that the
    proximal operator requires the solution of an inverse problem. If the
    operator ``Op`` is of kind ``explicit=True``, we can solve this problem
    directly. On the other hand if ``Op`` is of kind ``explicit=False``, an
    iterative solver is employed. In this case it is possible to provide a warm
    start via the ``x0`` input parameter.

    When only ``b`` is provided, ``Op`` is assumed to be an Identity operator
    and the proximal operator reduces to:

    .. math::

        \prox_{\tau f_\alpha}(\mathbf{x}) =
        \frac{\mathbf{x} + \tau \sigma \mathbf{b} - \tau \alpha \mathbf{q}}
        {1 + \tau \sigma}

    If ``b`` is not provided, the proximal operator reduces to:

    .. math::

        \prox_{\tau f_\alpha}(\mathbf{x}) =
        \frac{\mathbf{x} - \tau \alpha \mathbf{q}}{1 + \tau \sigma}

    Finally, note that the second term in :math:`f_\alpha(\mathbf{x})` is added
    because this combined expression appears in several problems where Bregman
    iterations are used alongside a proximal solver.

    """
    def __init__(self, dims, Op=None, b=None, q=None, sigma=1., alpha=1.,
                 lamda=1., gamma=.5, qgrad=True, niter=10, rtol=1e-4, 
                 x0=None, warm=True, densesolver=None, kwargs_solver=None):
        super().__init__(Op, True)
        self.dims = dims
        self.ndim = len(dims)
        self.b = b
        self.q = q
        self.sigma = sigma
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.qgrad = qgrad
        self.niter = niter
        self.rtol = rtol
        self.x0 = x0
        self.warm = warm
        self.densesolver = densesolver
        self.count = 0
        self.kwargs_solver = {} if kwargs_solver is None else kwargs_solver

        # when using factorize, store the first tau*sigma=0 so that the
        # first time it will be recomputed (as tau cannot be 0)
        if self.densesolver == 'factorize':
            self.tausigma = 0

        # create data term
        if self.Op is not None and self.b is not None:
            self.OpTb = self.sigma * self.Op.H @ self.b
            # create A.T A upfront for explicit operators
            if self.Op.explicit:
                self.ATA = np.conj(self.Op.A.T) @ self.Op.A

        # self.L2 = pyproximal.L2(Op, b, sigma, alpha, qgrad, niter, x0, 
        #                         warm, densesolver, kwargs_solver)
        self.isotropic_tv = pyproximal.TV(dims, 1., niter, rtol)


    def __call__(self, x):
        moreau_prox = self.isotropic_tv.prox(x, self.gamma)
        moreau_env = self.isotropic_tv.__call__(moreau_prox)
        if self.Op is not None and self.b is not None:
            f = (self.sigma / 2.) * (np.linalg.norm(self.Op * x - self.b) ** 2)
        elif self.b is not None:
            f = (self.sigma / 2.) * (np.linalg.norm(x - self.b) ** 2)
        else:
            f = (self.sigma / 2.) * (np.linalg.norm(x) ** 2)
        if self.q is not None:
            f += self.alpha * np.dot(self.q, x)
        return f - self.lamda * moreau_env
    
    def _increment_count(func):
        """Increment counter
        """
        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        # define current number of iterations
        if isinstance(self.niter, int):
            niter = self.niter
        else:
            niter = self.niter(self.count)

        # solve proximal optimization
        x += tau * self.lamda / self.gamma * (x - self.isotropic_tv.prox(x, self.gamma))
        if self.Op is not None and self.b is not None:
            y = x + tau * self.OpTb
            if self.q is not None:
                y -= tau * self.alpha * self.q
            if self.Op.explicit:
                if self.densesolver != 'factorize':
                    Op1 = MatrixMult(np.eye(self.Op.shape[1]) +
                                     tau * self.sigma * self.ATA)
                    if self.densesolver is None:
                        # to allow backward compatibility with PyLops versions earlier
                        # than v1.18.1 and v2.0.0
                        x = Op1.div(y)
                    else:
                        x = Op1.div(y, densesolver=self.densesolver)
                else:
                    if self.tausigma != tau * self.sigma:
                        # recompute factorization
                        self.tausigma = tau * self.sigma
                        ATA = np.eye(self.Op.shape[1]) + \
                              self.tausigma * self.ATA
                        self.cl = cho_factor(ATA)
                    x = cho_solve(self.cl, y)
            else:
                Op1 = Identity(self.Op.shape[1], dtype=self.Op.dtype) + \
                      float(tau * self.sigma) * (self.Op.H * self.Op)
                if get_module_name(get_array_module(x)) == 'numpy':
                    x = sp_lsqr(Op1, y, iter_lim=niter, x0=self.x0,
                                **self.kwargs_solver)[0]
                else:
                    x = lsqr(Op1, y, niter=niter, x0=self.x0,
                             **self.kwargs_solver)[0].ravel()
            if self.warm:
                self.x0 = x
        elif self.b is not None:
            num = x + tau * self.sigma * self.b
            if self.q is not None:
                num -= tau * self.alpha * self.q
            x = num / (1. + tau * self.sigma)
        else:
            num = x
            if self.q is not None:
                num -= tau * self.alpha * self.q
            x = num / (1. + tau * self.sigma)
        return x

    
    def grad(self, x):
        if self.Op is not None and self.b is not None:
            g = self.sigma * self.Op.H @ (self.Op @ x - self.b)
        elif self.b is not None:
            g = self.sigma * (x - self.b)
        else:
            g = self.sigma * x
        if self.q is not None and self.qgrad:
            g += self.alpha * self.q
        return g - self.lamda / self.gamma * (x - self.isotropic_tv.prox(x, self.gamma))
    


def UnadjustedLangevinPrimalDual(proxf, proxg, A, x0, tau, mu, y0=None, z=None, 
                                 theta=1., niter=10, seed=0, gfirst=True, callback=None, 
                                 callbacky=False, returny=False, show=False):
    r"""Unadjusted Langevin Primal-Dual algorithm (ULPDA)

    Solves the following (possibly) nonlinear minimization problem using
    the general version of the first-order primal-dual algorithm of [1]_:

    .. math::

        \min_{\mathbf{x} \in X} g(\mathbf{Ax}) + f(\mathbf{x}) +
        \mathbf{z}^T \mathbf{x}

    where :math:`\mathbf{A}` is a linear operator, :math:`f`
    and :math:`g` can be any convex functions that have a known proximal
    operator.

    This functional is effectively minimized by solving its equivalent
    primal-dual problem (primal in :math:`f`, dual in :math:`g`):

    .. math::

        \min_{\mathbf{x} \in X} \max_{\mathbf{y} \in Y}
        \mathbf{y}^T(\mathbf{Ax}) + \mathbf{z}^T \mathbf{x} +
        f(\mathbf{x}) - g^*(\mathbf{y})

    where :math:`\mathbf{y}` is the so-called dual variable.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator of g
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float` or :obj:`np.ndarray`
        Stepsize of subgradient of :math:`f`. This can be constant 
        or function of iterations (in the latter cases provided as np.ndarray)
    mu : :obj:`float` or :obj:`np.ndarray`
        Stepsize of subgradient of :math:`g^*`. This can be constant 
        or function of iterations (in the latter cases provided as np.ndarray)
    z0 : :obj:`numpy.ndarray`
        Initial auxiliary vector
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    theta : :obj:`float`
        Scalar between 0 and 1 that defines the update of the
        :math:`\bar{\mathbf{x}}` variable - note that ``theta=0`` is a
        special case that represents the semi-implicit classical Arrow-Hurwicz
        algorithm
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbacky : :obj:`bool`, optional
        Modify callback signature to (``callback(x, y)``) when ``callbacky=True``
    returny : :obj:`bool`, optional
        Return also ``y``
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Primal-dual algorithm can be expressed by the following recursion
    (``gfirst=True``):

    .. math::

        \mathbf{y}^{k+1} = \prox_{\mu g^*}(\mathbf{y}^{k} +
        \mu \mathbf{A}\bar{\mathbf{x}}^{k})\\
        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^{k} -
        \tau (\mathbf{A}^H \mathbf{y}^{k+1} + \mathbf{z})) \\
        \bar{\mathbf{x}}^{k+1} = \mathbf{x}^{k+1} +
        \theta (\mathbf{x}^{k+1} - \mathbf{x}^k)

    where :math:`\tau \mu \lambda_{max}(\mathbf{A}^H\mathbf{A}) < 1`.

    Alternatively for ``gfirst=False`` the scheme becomes:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^{k} -
        \tau (\mathbf{A}^H \mathbf{y}^{k} + \mathbf{z})) \\
        \bar{\mathbf{x}}^{k+1} = \mathbf{x}^{k+1} +
        \theta (\mathbf{x}^{k+1} - \mathbf{x}^k) \\
        \mathbf{y}^{k+1} = \prox_{\mu g^*}(\mathbf{y}^{k} +
        \mu \mathbf{A}\bar{\mathbf{x}}^{k+1})

    .. [1] A., Chambolle, and T., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120-145. 2011.

    """
    ncp = get_array_module(x0)

    # check if tau and mu are scalars or arrays
    fixedtau = fixedmu = False
    if isinstance(tau, (int, float)):
        tau = tau * ncp.ones(niter, dtype=x0.dtype)
        fixedtau = True
    if isinstance(mu, (int, float)):
        mu = mu * ncp.ones(niter, dtype=x0.dtype)
        fixedmu = True

    if show:
        tstart = time.time()
        print('Unadjusted Langevin primal-dual: U(x) = f(Ax) + x^T z + g(x)\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'Additional vector (z): %s\n'
              'tau = %s\t\tmu = %s\ntheta = %.2f\t\tniter = %d\n' %
              (type(proxf), type(proxg), type(A),
               None if z is None else 'vector', str(tau[0]) if fixedtau else 'Variable',
               str(mu[0]) if fixedmu else 'Variable', theta, niter))
        head = '   Itn       x[0]          f           g          z^x       J = f + g + z^x'
        print(head)

    x = x0.copy()
    xhat = x.copy()
    y = y0.copy() if y0 is not None else ncp.zeros(A.shape[0], dtype=x.dtype)
    x_samples = []
    y_samples = []
    rng = default_rng(seed)
    for iiter in range(niter):
        xi = scipy.stats.multivariate_normal.rvs(size=x.shape, random_state=rng)
        xold = x.copy()
        if gfirst:
            y = proxg.proxdual(y + mu[iiter] * A.matvec(xhat), mu[iiter])
            ATy = A.rmatvec(y)
            if z is not None:
                ATy += z
            x = proxf.prox(x - tau[iiter] * ATy, tau[iiter]) + np.sqrt(2 * tau[iiter]) * xi
            xhat = x + theta * (x - xold)
        else:
            ATy = A.rmatvec(y)
            if z is not None:
                ATy += z
            x = proxf.prox(x - tau[iiter] * ATy, tau[iiter]) + np.sqrt(2 * tau[iiter]) * xi
            xhat = x + theta * (x - xold)
            y = proxg.proxdual(y + mu[iiter] * A.matvec(xhat), mu[iiter])
        x_samples.append(x)
        y_samples.append(y)

        # run callback
        if callback is not None:
            if callbacky:
                callback(x, y)
            else:
                callback(x)
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(A.matvec(x))
                pf = 0. if type(pf) == bool else pf
                pg = 0. if type(pg) == bool else pg
                zx = 0. if z is None else np.dot(z, x)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e      %10.3e' % \
                      (iiter + 1, x[0], pf, pg, zx, pf + pg + zx)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    if not returny:
        return np.array(x_samples)
    else:
        return np.array(x_samples), np.array(y_samples)



def prox_conjugate(x, gamma, prox):
    return x - gamma * prox(x / gamma, 1/gamma)


def prox_square_loss(x, y, H, gamma):
    d = x.shape[0] * x.shape[1]
    return (pylops.Identity(d) + gamma * H.adjoint() * H).div(x + gamma * H.adjoint() * y)


def prox_laplace(x, gamma): 
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def prox_uncentered_laplace(x, gamma, mu):
    return mu + prox_laplace(x - mu, gamma)


def prox_gaussian(x, gamma):
    return x / (2*gamma + 1)


def prox_gen_gaussian(x, gamma, p):
    if p == 4/3:
        xi = np.sqrt(x**2 + 256*gamma**3/729)
        prox = x + 4 * gamma / (3*2**(1/3)) * ((xi - x)**(1/3) - (xi + x)**(1/3))
    elif p == 3/2: 
        prox = x + 9 * gamma**2 * np.sign(x) * (1 - np.sqrt(1 + 16 * np.abs(x)/(9*gamma**2))) / 8
    elif p == 3:
        prox = np.sign(x) * (np.sqrt(1 + 12*gamma*np.abs(x)) - 1) / (6*gamma)
    elif p == 4:
        xi = np.sqrt(x**2 + 1/(27*gamma))
        prox = ((xi + x)/(8*gamma))**(1/3) - ((xi - x)/(8*gamma))**(1/3)
    return prox


def prox_huber(x, gamma, tau):    
    return x / (2*tau + 1) if np.abs(x) <= gamma * (2*tau + 1) / np.sqrt(2*tau) else x - gamma * np.sqrt(2 * tau) * np.sign(x)
    

def prox_max_ent(x, gamma, tau, kappa, p):
    return np.sign(x) * prox_gen_gaussian(1/(2*tau+1) * np.maximum(np.abs(x) - gamma, 0), kappa/(2*tau+1))


def prox_smoothed_laplace(x, gamma):
    return np.sign(x) * (gamma * np.abs(x) - gamma**2 - 1 + np.sqrt(np.abs(gamma*np.abs(x) - gamma**2 - 1)**2 + 4*gamma*np.abs(x))) / (2*gamma)


def prox_exp(x, gamma): 
    return x - gamma if x >= gamma else 0.


def prox_gamma(x, omega, kappa):
    return (x - omega + np.sqrt((x - omega)**2 + 4*kappa)) / 2


def prox_chi(x, kappa):
    return (x + np.sqrt(x**2 + 8*kappa)) / 4


def prox_uniform(x, omega):
    if x < -omega:
        p = -omega
    elif x > omega:
        p = omega
    else: 
        p = x
    return p


def prox_triangular(x, omega1, omega2):
    if x < 1 / omega1:
        p = (x + omega1 + np.sqrt((x - omega1)**2 + 4)) / 2
    elif x > 1 / omega2:
        p = (x + omega2 + np.sqrt((x - omega2)**2 + 4)) / 2
    else: 
        p = 0.
    return p


def prox_weibull(x, omega, kappa, p):
    f = lambda y: p * omega * y**p + y**2 - x * y - kappa
    res = minimize_scalar(f, bounds=(0, np.inf), method='bounded')
    return res.x


def prox_gen_inv_gaussian(x, omega, kappa, rho):
    f = lambda y: y**3 + (omega - x) * y**2 - kappa * y - rho
    res = minimize_scalar(f, bounds=(0, np.inf), method='bounded')
    return res.x


def prox_pearson_I(x, kappa1, kappa2, omega1, omega2):
    f = lambda y: y**3 - (omega1 + omega2 + x) * y**2 + (omega1*omega2 - kappa1 - kappa2 + (omega1 + omega2)*x)*y \
        - omega1*omega2*x + omega1*kappa2 + omega2*kappa1
    res = minimize_scalar(f, bounds=(omega1, omega2), method='bounded')
    return res.x


def prox_tv(x, gamma):

    return 

