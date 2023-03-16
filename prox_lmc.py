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

'''
Usage: python prox_lmc.py --gamma_pgld=5e-2 --gamma_myula=5e-2 --gamma_mymala=5e-2 \
--gamma_ppula=5e-2 --gamma_fbula=5e-2 --gamma_lbmumla=5e-2 --gamma0_ulpda=5e-2 \
--gamma1_ulpda=5e-2 --alpha=1.5e-1 --lamda=2.5e-1 --t=100 --seed=0 --K=10000 --n=5
'''

import os
from fastprogress import progress_bar
import fire

import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import scienceplots
plt.style.use(['science'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    } 
    )

from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from scipy.integrate import quad, dblquad

from prox import *


class ProximalLangevinMonteCarlo:
    def __init__(self, mus, Sigmas, omegas, lamda, alpha, K=1000, seed=0) -> None:
        super(ProximalLangevinMonteCarlo, self).__init__()
        self.mus = mus
        self.Sigmas = Sigmas
        self.omegas = omegas
        self.lamda = lamda
        self.alpha = alpha
        self.n = K
        self.seed = seed
        self.d = mus[0].shape[0]

    def multivariate_gaussian(self, theta, mu, Sigma):
        """Return the multivariate Gaussian distribution on array theta."""
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**self.d * np.abs(Sigma_det))
        # This einsum call calculates (theta - mu)T.Sigma-1.(theta - mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', theta - mu, Sigma_inv, theta - mu)
        return np.exp(-fac / 2) / N

    def density_gaussian_mixture(self, theta): 
        den = [self.omegas[i] * self.multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(den)

    def potential_gaussian_mixture(self, theta): 
        return -np.log(self.density_gaussian_mixture(theta))

    def laplacian_prior(self, theta):    
        d = theta.shape[0]
        return (self.alpha/2)**d * np.exp(-self.alpha * np.linalg.norm(theta, ord=1, axis=-1))

    def grad_density_multivariate_gaussian(self, theta, mu, Sigma):
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**self.d * np.abs(Sigma_det))
        fac = np.einsum('...k,kl,...l->...', theta - mu, Sigma_inv, theta - mu)    
        return np.exp(-fac / 2) / N * Sigma_inv @ (mu - theta)

    def grad_density_gaussian_mixture(self, theta):
        grad_den = [self.omegas[i] * self.grad_density_multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(grad_den)

    def grad_potential_gaussian_mixture(self, theta):
        return -self.grad_density_gaussian_mixture(theta) / self.density_gaussian_mixture(theta)

    def hess_density_multivariate_gaussian(self, theta, mu, Sigma):
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**self.d * np.abs(Sigma_det))
        fac = np.einsum('...k,kl,...l->...', theta - mu, Sigma_inv, theta - mu)
        return np.exp(-fac / 2) / N * (Sigma_inv @ np.outer(theta - mu, theta - mu) @ Sigma_inv - Sigma_inv)

    def hess_density_gaussian_mixture(self, theta):
        hess_den = [self.omegas[i] * self.hess_density_multivariate_gaussian(theta, self.mus[i], self.Sigmas[i]) for i in range(len(self.mus))]
        return sum(hess_den)
        
    def hess_potential_gaussian_mixture(self, theta):
        density = self.density_gaussian_mixture(theta)
        grad_density = self.grad_density_gaussian_mixture(theta)
        hess_density = self.hess_density_gaussian_mixture(theta)
        return np.outer(grad_density, grad_density) / density**2 - hess_density / density

    def gd_update(self, theta, gamma): 
        return theta - gamma * self.grad_potential_gaussian_mixture(theta) 


    ## Proximal Gradient Langevin Dynamics (PGLD)
    def pgld(self, gamma):
        print("\nSampling with Proximal ULA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        for _ in progress_bar(range(self.n)):        
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta0 = prox_laplace(theta0, self.lamda * self.alpha)
            theta_new = self.gd_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)
    

    ## Moreau--Yosida Unadjusted Langevin Algorithm (MYULA)
    def grad_Moreau_env(self, theta):
        return (theta - prox_laplace(theta, self.lamda * self.alpha)) / self.lamda

    def prox_update(self, theta, gamma):
        return -gamma * self.grad_Moreau_env(theta)

    def myula(self, gamma):
        print("\nSampling with MYULA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = self.gd_update(theta0, gamma) + self.prox_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Moreau--Yosida regularized Metropolis-Adjusted Langevin Algorithm (MYMALA)
    def q_prob(self, theta1, theta2, gamma):
        return multivariate_normal(mean=self.gd_update(theta2, gamma) + self.prox_update(theta2, gamma), cov=2*gamma).pdf(theta1)


    def prob(self, theta_new, theta_old, gamma):
        density_ratio = ((self.density_gaussian_mixture(theta_new) * self.laplacian_prior(theta_new)) / 
                        (self.density_gaussian_mixture(theta_old) * self.laplacian_prior(theta_old)))
        q_ratio = self.q_prob(theta_old, theta_new, gamma) / self.q_prob(theta_new, theta_old, gamma)
        return density_ratio * q_ratio


    def mymala(self, gamma):
        print("\nSampling with MYMALA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = self.gd_update(theta0, gamma) + self.prox_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            p = self.prob(theta_new, theta0, gamma)
            alpha = min(1, p)
            if rng.random() <= alpha:
                theta.append(theta_new) 
                theta0 = theta_new
        return np.array(theta), len(theta)


    ## Preconditioned Proximal ULA 
    def preconditioned_gd_update(self, theta, gamma, M): 
        return theta - gamma * M @ self.grad_potential_gaussian_mixture(theta)

    def preconditioned_prox(self, x, gamma, M, t=100): 
        rho = 1 / np.linalg.norm(M, ord=2)
        eps = min(1, rho)
        eta = 2 * rho - eps
        w = np.zeros(x.shape[0])
        for _ in range(t):
            u = x - M @ w
            w += eta * u - eta * prox_laplace(eta*w + u, gamma / eta)
        return w

    def preconditioned_prox_update(self, theta, gamma, M, t=100):
        return self.preconditioned_prox(theta - gamma * M @ self.grad_density_gaussian_mixture(theta), gamma, M, t)

    def ppula(self, gamma, M, t=100):
        print("\nSampling with PP-ULA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = self.preconditioned_prox_update(theta0, gamma, M, t) + np.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Forward-Backward Unadjusted Langevin Algorithm (FBULA)
    def grad_FB_env(self, theta):
        return (np.eye(theta.shape[0]) - self.lamda * self.hess_potential_gaussian_mixture(theta)) @ (theta - prox_laplace(self.gd_update(theta, self.lamda), self.lamda * self.alpha)) / self.lamda

    def gd_FB_update(self, theta, gamma):
        return theta - gamma * self.grad_FB_env(theta)

    def fbula(self, gamma):
        print("\nSampling with EULA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = self.gd_FB_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Bregman--Moreau Unadjusted Mirror-Langevin Algorithm (BMUMLA)
    def grad_mirror_hyp(self, theta, beta): 
        return np.arcsinh(theta / beta)

    def grad_conjugate_mirror_hyp(self, theta, beta):
        return beta * np.sinh(theta)

    def left_bregman_prox_ell_one_hypent(self, theta, beta, gamma):
        if isinstance(theta, float):
            if theta > beta * np.sinh(gamma):
                prox = beta * np.sinh(np.arcsinh(theta / beta) - gamma)
            elif theta < beta * np.sinh(-gamma):
                prox = beta * np.sinh(np.arcsinh(theta / beta) + gamma)
            else: 
                prox = np.sqrt(theta ** 2 + beta ** 2) - beta
        else:
            prox = np.array(len(theta))
            p1 = beta * np.sinh(np.arcsinh(theta / beta) - gamma)
            p2 = beta * np.sinh(np.arcsinh(theta / beta) + gamma)
            p3 = np.sqrt(theta ** 2 + beta ** 2) - beta
            prox = np.where(theta > beta * np.sinh(gamma), p1, p3)
            prox = np.where(theta < beta * np.sinh(-gamma), p2, prox)
        return prox

    def grad_BM_env(self, theta, beta):
        return 1/self.lamda * (theta**2 + beta**2)**(-.5) * (theta - self.left_bregman_prox_ell_one_hypent(theta, beta, self.lamda * self.alpha))

    def gd_BM_update(self, theta, gamma):
        return -gamma * self.grad_potential_gaussian_mixture(theta)

    def prox_BM_update(self, theta, beta, gamma):
        return -gamma * self.grad_BM_env(theta, beta)

    def lbmumla(self, gamma, beta, sigma):
        print("\nSampling with LBMUMLA: ")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = self.grad_mirror_hyp(theta0, beta) + self.gd_BM_update(theta0, gamma) + self.prox_BM_update(theta0, sigma, gamma) + np.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
            theta_new = self.grad_conjugate_mirror_hyp(theta_new, beta)
            theta.append(theta_new) 
            theta_new = self.grad_mirror_hyp(theta0, beta) - gamma *  + np.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
            theta0 = theta_new
        return np.array(theta)


    # Unadjusted Langevin Primal-Dual Algorithm (ULPDA)
    def ulpda(self, gamma0, gamma1, tau, D, prox_f, prox_g):
        print("\nSampling with Unadjusted Langevin Primal-Dual Algorithm (ULPDA):")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        u0 = tu0 = rng.normal(0, 1, self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = prox_f(theta0 - gamma0 * D.T @ tu0, gamma0) + np.sqrt(2*gamma0) * xi
            u_new = prox_conjugate(u0 + gamma1 * D @ (2*theta_new - theta0), gamma1, prox_g)
            tu_new = u0 + tau * (u_new - u0)
            theta.append(theta_new)    
            theta0 = theta_new
            u0 = u_new
            tu0 = tu_new
        return np.array(theta)


    '''
    def error(self, thetas1, thetas2):
        density_2d_gaussian_mixture(theta, mus, Sigmas)
        np.sum()

        return 
    '''


## Main function
def prox_lmc_gaussian_mixture(gamma_pgld=5e-2, gamma_myula=5e-2, 
                                gamma_mymala=5e-2, gamma_ppula=5e-2, 
                                gamma_fbula=5e-2, gamma_lbmumla=5e-2,
                                gamma0_ulpda=5e-2, gamma1_ulpda=5e-2, 
                                lamda=0.01, alpha=.1, n=5, t=100, 
                                K=10000, seed=0):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 100
    X = np.linspace(-8, 8, N)
    Y = np.linspace(-8, 8, N)
    X, Y = np.meshgrid(X, Y)


    # Mean vectors and covariance matrices
    mu1 = np.array([3., 3.])
    Sigma1 = np.array([[ 1. , -0.5], [-0.5,  1.]])
    mu2 = np.array([-4., 6.])
    Sigma2 = np.array([[0.5, 0.2], [0.2, 0.7]])
    mu3 = np.array([4., -6.])
    Sigma3 = np.array([[0.5, 0.1], [0.1, 0.9]])
    mu4 = np.array([5., 5.])
    Sigma4 = np.array([[0.8, 0.02], [0.02, 0.3]])
    mu5 = np.array([-4., -4.])
    Sigma5 = np.array([[1.2, 0.05], [0.05, 0.8]])


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
    omegas = np.ones(n) / n

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    prox_lmc = ProximalLangevinMonteCarlo(mus, Sigmas, omegas, lamda, alpha, K, seed)

    # The distribution on the variables X, Y packed into pos.
    Z = prox_lmc.density_gaussian_mixture(pos) * prox_lmc.laplacian_prior(pos)

    ## Plot of the true Gaussian mixture
    print("\nPlotting the true Gaussian mixture...")
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax1.view_init(45, -70)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    # plt.suptitle("True 2D Gaussian Mixture") 
    # plt.show()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    fig.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_1.pdf', dpi=500)

    Z1 = prox_lmc.pgld(gamma_pgld)

    Z2 = prox_lmc.myula(gamma_myula)

    Z3, eff_K = prox_lmc.mymala(gamma_mymala)
    print(f'\nMYMALA acceptance rate: {eff_K / K} ')

    M = np.array([[1.0, 0.1], [0.1, 0.5]])
    Z4 = prox_lmc.ppula(gamma_ppula, M, t)

    Z5 = prox_lmc.fbula(gamma_fbula)
    
    # beta = np.array([0.2, 0.8])
    beta = np.array([0.7, 0.3])
    sigma = np.array([0.2, 0.8])
    Z6 = prox_lmc.lbmumla(gamma_lbmumla, beta, sigma)

    D = np.eye(2)
    tau = .5
    Z7 = prox_lmc.ulpda(gamma0_ulpda, gamma1_ulpda, tau, D, prox_gaussian, prox_laplace)


    ## Plot of the true Gaussian mixture with 2d histograms of samples
    print("\nConstructing the 2D histograms of samples...")
    fig3, axes = plt.subplots(2, 4, figsize=(17, 8))

    # axes[0,0].hist2d(Z[:, 0], Z[:, 1], bins=100, density=True)
    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].hist2d(Z1[:,0], Z1[:,1], bins=100, cmap=cm.viridis)
    # sns.kdeplot(x=Z1[:,0], y=Z1[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,1].set_title("PGLD", fontsize=16)

    axes[0,2].hist2d(Z2[:,0], Z2[:,1], bins=100, cmap=cm.viridis)
    # sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("MYULA", fontsize=16)

    axes[0,3].hist2d(Z3[:,0], Z3[:,1], bins=100, cmap=cm.viridis)
    # sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,3])
    axes[0,3].set_title("PP-ULA", fontsize=16)

    axes[1,0].hist2d(Z4[:,0], Z4[:,1], bins=100, cmap=cm.viridis)
    # sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("MYMALA", fontsize=16)

    axes[1,1].hist2d(Z5[:,0], Z5[:,1], bins=100, cmap=cm.viridis)
    # sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("FBULA", fontsize=16)

    axes[1,2].hist2d(Z6[:,0], Z6[:,1], bins=100, cmap=cm.viridis)
    # sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("LBMUMLA", fontsize=16)

    axes[1,3].hist2d(Z7[:,0], Z7[:,1], bins=100, cmap=cm.viridis)
    # sns.kdeplot(x=Z7[:,0], y=Z7[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,3])
    axes[1,3].set_title("ULPDA", fontsize=16)

    plt.show(block=False)
    plt.pause(10)
    plt.close()
    fig3.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_3.pdf', dpi=500)

    
    ## Plot of the true Gaussian mixture with KDE of samples
    print("\nConstructing the KDEs of samples...")
    # fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig2, axes = plt.subplots(2, 4, figsize=(17, 8))
    # fig2.suptitle("True density and KDEs of samples") 

    sns.set(font='serif', rc={'figure.figsize':(3.25, 3.5)})

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    sns.kdeplot(x=Z1[:,0], y=Z1[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,1].set_title("PGLD", fontsize=16)

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("MYULA", fontsize=16)

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,3])
    axes[0,3].set_title("PP-ULA", fontsize=16)

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("MYMALA", fontsize=16)

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("FBULA", fontsize=16)

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("LBMUMLA", fontsize=16)

    sns.kdeplot(x=Z7[:,0], y=Z7[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,3])
    axes[1,3].set_title("ULPDA", fontsize=16)
    
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    fig2.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_2.pdf', dpi=500)  
    

if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(prox_lmc_gaussian_mixture)