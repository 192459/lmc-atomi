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

# Install libraries: pip install -U numpy matplotlib scipy seaborn fire

# To install JAX, see its documentations


'''
Usage: python lmc_jax.py --gamma_ula=5e-2 --gamma_mala=5e-2 --gamma_pula=5e-2 --gamma_ihpula=5e-2 --gamma_mla=5e-2 --nChains=4 --K=10000 --n=5 --seed=0
'''

import os
import itertools
from fastprogress import progress_bar
from typing import NamedTuple
import multiprocessing
import fire

import numpy as np
import random

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())
import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    } 
    )

if jax.lib.xla_bridge.get_backend().platform == "cpu":
    from jax.scipy.linalg import sqrtm
else:
    def sqrtm(x):
        u, s, vh = jnp.linalg.svd(x)
        return (u * s**.5) @ vh

class LangevinMonteCarlo:
    def __init__(self, mus, Sigmas, omegas, nChains=4, K=1000, seed=0) -> None:
        super(LangevinMonteCarlo, self).__init__()
        self.mus = mus
        self.Sigmas = Sigmas
        self.omegas = omegas
        self.nChains = nChains
        self.n = K
        self.seed = seed
        self.d = mus[0].shape[0]

    def multivariate_gaussian(self, theta, mu, Sigma):
        return stats.multivariate_normal.pdf(theta, mu, Sigma)

    def density_gaussian_mixture(self, theta): 
        den = [self.omegas[i] * self.multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(den)

    def potential_gaussian_mixture(self, theta): 
        return -jnp.log(self.density_gaussian_mixture(theta))


    def grad_potential_gaussian_mixture(self, theta): 
        return jax.grad(self.potential_gaussian_mixture, 0)

    # '''
    def grad_density_multivariate_gaussian(self, pos, mu, Sigma):
        n = mu.shape[0]
        Sigma_det = jnp.linalg.det(Sigma)
        Sigma_inv = jnp.linalg.inv(Sigma)
        N = jnp.sqrt((2*jnp.pi)**n * np.abs(Sigma_det))
        fac = jnp.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)    
        return jnp.exp(-fac / 2) / N * Sigma_inv @ (mu - pos)


    def grad_density_gaussian_mixture(self, theta, mus, Sigmas, lambdas):
        K = len(mus)
        grad_den = [lambdas[k] * self.grad_density_multivariate_gaussian(theta, mus[k], Sigmas[k]) for k in range(K)]
        return sum(grad_den)


    def grad_potential_gaussian_mixture(self, theta, mus, Sigmas, lambdas):
        return -self.grad_density_gaussian_mixture(theta, mus, Sigmas, lambdas) / self.density_gaussian_mixture(theta, mus, Sigmas, lambdas)
    # '''

    # hess_potential_2d_gaussian_mixture = jax.hessian(potential_2d_gaussian_mixture, 0)

    # '''
    def hess_density_multivariate_gaussian(self, pos, mu, Sigma):
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * np.abs(Sigma_det))
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        return np.exp(-fac / 2) / N * (Sigma_inv @ np.outer(pos - mu, pos - mu) @ Sigma_inv - Sigma_inv)


    def hess_density_gaussian_mixture(self, theta, mus, Sigmas, lambdas):
        K = len(mus)
        hess_den = [lambdas[k] * self.hess_density_multivariate_gaussian(theta, mus[k], Sigmas[k]) for k in range(K)]
        return sum(hess_den)
        

    def hess_potential_gaussian_mixture(self, theta, mus, Sigmas, lambdas):
        density = self.density_gaussian_mixture(theta, mus, Sigmas, lambdas)
        grad_density = self.grad_density_gaussian_mixture(theta, mus, Sigmas, lambdas)
        hess_density = self.hess_density_gaussian_mixture(theta, mus, Sigmas, lambdas)
        return np.outer(grad_density, grad_density) / density**2 - hess_density / density
    # '''

    # @jax.jit
    def gd_update(self, theta, mus, Sigmas, lambdas, gamma): 
        return theta - gamma * self.grad_potential_gaussian_mixture(theta, mus, Sigmas, lambdas)


    ## Unadjusted Langevin Algorithm (ULA)
    # Use jax.pmap to sample with multiple chains
    def ula_gaussian_mixture(self, gamma, mus, Sigmas, lambdas, d=2, n=1000, seed=0):
        key = jax.random.PRNGKey(seed)
        theta0 = jax.random.normal(key, (d,))
        theta = []
        for _ in range(n):
            xi = jax.random.multivariate_normal(key, jnp.zeros(d), jnp.eye(d))
            theta_new = self.gd_update(theta0, mus, Sigmas, lambdas, gamma) + jnp.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return jnp.array(theta)


    ## Metropolis-Adjusted Langevin Algorithm (MALA)
    def q_prob(self, theta1, theta2, gamma, mus, Sigmas, lambdas):
        return jnp.exp(-1/(4*gamma) * jnp.linalg.norm(theta1 - self.gd_update(theta2, mus, Sigmas, lambdas, gamma))**2)


    def prob(self, theta_new, theta_old, gamma, mus, Sigmas, lambdas):
        density_ratio = self.density_gaussian_mixture(theta_new, mus, Sigmas, lambdas) / self.density_gaussian_mixture(theta_old, mus, Sigmas, lambdas)
        q_ratio = self.q_prob(theta_old, theta_new, gamma, mus, Sigmas, lambdas) / self.q_prob(theta_new, theta_old, gamma, mus, Sigmas, lambdas)
        return density_ratio * q_ratio


    def mala_gaussian_mixture(self, gamma, mus, Sigmas, lambdas, d=2, n=1000, seed=0):
        key = jax.random.PRNGKey(seed)
        theta0 = jax.random.normal(key, (d,))
        theta = []
        for _ in range(n):
            xi = jax.random.multivariate_normal(key, jnp.zeros(d), jnp.eye(d))
            theta_new = self.gd_update(theta0, mus, Sigmas, lambdas, gamma) + jnp.sqrt(2*gamma) * xi
            p = self.prob(theta_new, theta0, gamma, mus, Sigmas, lambdas)
            alpha = min(1, p)
            if random.random() <= alpha:
                theta.append(theta_new)    
                theta0 = theta_new
        return jnp.array(theta), len(theta)


    ## Preconditioned ULA 
    # @jax.jit
    def preconditioned_gd_update(self, theta, mus, Sigmas, lambdas, gamma, M): 
        return theta - gamma * M @ self.grad_potential_gaussian_mixture(theta, mus, Sigmas, lambdas)

    # @jax.jit
    # def hess_preconditioned_gd_update(theta, mus, Sigmas, lambdas, gamma): 
    #     hess = hess_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)
    #     eig = np.linalg.eig(hess)
    #     M = hess + eig * np.eye(hess.shape[0])
    #     return theta - gamma * np.linalg.inv(M) @ grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)   


    def preconditioned_langevin_gaussian_mixture(self, gamma, mus, Sigmas, lambdas, M, d=2, n=1000, seed=0):
        key = jax.random.PRNGKey(seed)
        theta0 = jax.random.normal(key, (d,))
        theta = []
        for _ in range(n):
            xi = jax.random.multivariate_normal(key, jnp.zeros(d), jnp.eye(d))
            theta_new = self.preconditioned_gd_update(theta0, mus, Sigmas, lambdas, gamma, M) + jnp.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return jnp.array(theta)


    # Preconditioning with inverse Hessian
    def hess_preconditioned_langevin_gaussian_mixture(self, gamma, mus, Sigmas, lambdas, d=2, n=1000, seed=0):
        key = jax.random.PRNGKey(seed)
        theta0 = jax.random.normal(key, (d,))
        theta = []    
        for _ in range(n):
            xi = jax.random.multivariate_normal(key, jnp.zeros(d), jnp.eye(d))
            hess = self.hess_potential_gaussian_mixture(theta0, mus, Sigmas, lambdas)
            # e, _ = np.linalg.eig(hess)
            # M = hess + np.abs(min(e)) * np.eye(hess.shape[0])
            # print(np.linalg.det(hess))
            M = jnp.linalg.inv(hess)
            theta_new = self.preconditioned_gd_update(theta0, mus, Sigmas, lambdas, gamma, M) + jnp.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return jnp.array(theta)


    ## Mirror-Langevin Algorithm (MLA)
    def grad_mirror_hyp(self, theta, beta): 
        return jnp.arcsinh(theta / beta)

    def grad_conjugate_mirror_hyp(self, theta, beta):
        return beta * jnp.sinh(theta)

    def mla_gaussian_mixture(self, gamma, mus, Sigmas, lambdas, beta, d=2, n=1000, seed=0):
        key = jax.random.PRNGKey(seed)
        theta0 = jax.random.normal(key, (d,))
        theta = []
        for _ in range(n):
            xi = jax.random.multivariate_normal(key, jnp.zeros(d), jnp.eye(d))
            theta_new = self.grad_mirror_hyp(theta0, beta) - gamma * self.grad_potential_gaussian_mixture(theta0, mus, Sigmas, lambdas) + jnp.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
            theta_new = self.grad_conjugate_mirror_hyp(theta_new, beta)
            theta.append(theta_new)    
            theta0 = theta_new
        return jnp.array(theta)



## Main function
def lmc_gaussian_mixture(gamma_ula=5e-2, gamma_mala=5e-2, 
                         gamma_pula=5e-2, gamma_ihpula=5e-2, 
                         gamma_mla=5e-2, nChains=4, n=5, K=5000, seed=0):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 100
    X = jnp.linspace(-5, 5, N)
    Y = jnp.linspace(-5, 5, N)
    X, Y = jnp.meshgrid(X, Y)

    # Mean vectors and covariance matrices
    mu1 = jnp.array([0., 0.])
    Sigma1 = jnp.array([[ 1. , -0.5], [-0.5,  1.]])
    mu2 = jnp.array([-2., 3.])
    Sigma2 = jnp.array([[0.5, 0.2], [0.2, 0.7]])
    mu3 = jnp.array([2., -3.])
    Sigma3 = jnp.array([[0.5, 0.1], [0.1, 0.9]])
    mu4 = jnp.array([3., 3.])
    Sigma4 = jnp.array([[0.8, 0.02], [0.02, 0.3]])
    mu5 = jnp.array([-2., -2.])
    Sigma5 = jnp.array([[1.2, 0.05], [0.05, 0.8]])


    if n == 1:
        mus = [mu1]
        Sigmas = [Sigma1]
    elif n == 2: 
        mus = [mu1, mu2]
        Sigmas = [Sigma1, Sigma2]
    elif n == 3: 
        mus = [mu1, mu2, mu3]
        Sigmas = [Sigma1, Sigma2, Sigma3]
    elif n == 4: 
        mus = [mu2, mu3, mu4, mu5]
        Sigmas = [Sigma2, Sigma3, Sigma4, Sigma5]
    elif n == 5: 
        mus = [mu1, mu2, mu3, mu4, mu5]
        Sigmas = [Sigma1, Sigma2, Sigma3, Sigma4, Sigma5]


    # weight vector
    omegas = jnp.ones(n) / n

    # Pack X and Y into a single 3-dimensional array
    # pos = jnp.empty(X.shape + (2,))
    # pos = pos.at[:, :, 0].set(X)
    # pos = pos.at[:, :, 1].set(Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    pos = jnp.array(pos)

    nChains = min(nChains, len(jax.devices()))

    lmc = LangevinMonteCarlo(mus, Sigmas, omegas, nChains, K, seed)

    # The distribution on the variables X, Y packed into pos.
    Z = lmc.density_gaussian_mixture(pos)
    # Z = np.exp(np.log(density_2d_gaussian_mixture(pos, mus, Sigmas, lambdas)) - 0.005 * np.sum(np.abs(pos)))


    ## Plot of the true Gaussian mixture
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax1.view_init(45, -70)
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.set_zticks([])
    # ax1.set_xlabel(r'$x_1$')
    # ax1.set_ylabel(r'$x_2$')


    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    # ax2.set_xlabel(r'$x_1$')
    # ax2.set_ylabel(r'$x_2$')

    # plt.suptitle("True 2D Gaussian Mixture") 
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig.savefig(f'./fig/fig_{n}_1.pdf', dpi=500)


    Z2 = lmc.ula_gaussian_mixture(gamma_ula, mus, Sigmas, omegas, n=K)

    sns.set(rc={'figure.figsize':(3.25, 3.5)})



    Z3, eff_K = lmc.mala_gaussian_mixture(gamma_mala, mus, Sigmas, omegas, n=K)
    print(f'\nMALA percentage of effective samples: {eff_K / K}')

    M = jnp.array([[1.0, 0.1], [0.1, 0.5]])
    Z4 = lmc.preconditioned_langevin_gaussian_mixture(gamma_pula, mus, Sigmas, omegas, M, n=K)

    Z5 = lmc.hess_preconditioned_langevin_gaussian_mixture(gamma_ihpula, mus, Sigmas, omegas, n=K)

    # beta = np.array([0.2, 0.8])
    beta = jnp.array([0.7, 0.3])
    Z6 = lmc.mla_gaussian_mixture(gamma_mla, mus, Sigmas, omegas, beta, n=K)



    ## Plot of the true Gaussian mixture with KDE of samples
    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    # fig2.suptitle("True density and KDEs of samples") 

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density")

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,1].set_title("ULA")

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("MALA")

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("PULA")

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("IHPULA")

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("MLA")

    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig2.savefig(f'./fig/fig_{n}_2.pdf', dpi=500)  




if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(lmc_gaussian_mixture)