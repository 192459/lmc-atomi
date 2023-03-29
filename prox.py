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

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
import pylops

from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr
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

        self.L2 = pyproximal.L2(Op, b, sigma, alpha, qgrad, niter, x0, 
                                warm, densesolver, kwargs_solver)
        self.isotropic_tv = pyproximal.TV(dims, 1., niter, rtol)


    def __call__(self, x):
        moreau_prox = self.isotropic_tv.prox(x, self.gamma)
        moreau_env = self.isotropic_tv.__call__(moreau_prox)
        return self.L2.__call__(x) - self.lamda * moreau_env
    
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
        x = self.L2.prox(x, tau)
        return x
    
    def grad(self, x):
        return self.L2.grad(x) - self.lamda / self.gamma * (x - self.isotropic_tv(x, self.gamma))
    

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

