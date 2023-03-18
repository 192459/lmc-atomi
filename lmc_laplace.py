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

# Install libraries: pip install -U numpy matplotlib scipy seaborn fire fastprogress

'''
Usage: 
python lmc_laplace.py --gamma_ula=5e-2 --gamma_mala=5e-2 --gamma_pula=5e-2 --gamma_ihpula=5e-4 --gamma_mla=5e-2 --lamda=1e0 --alpha=5e-1 --n=5 --K=10000 --seed=0
'''

import os
from fastprogress import progress_bar
import fire

import numpy as np
from numpy.random import default_rng
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

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

from prox import *

class LangevinMonteCarloLaplacian:
    def __init__(self, mus, alphas, omegas, lamda, K=1000, seed=0) -> None:
        super(LangevinMonteCarloLaplacian, self).__init__()
        self.mus = mus
        self.alphas = alphas
        self.omegas = omegas
        self.lamda = lamda        
        self.n = K
        self.seed = seed
        self.d = mus[0].shape[0]  

    def multivariate_laplacian(self, theta, mu, alpha):
        return (alpha/2)**self.d * np.exp(-alpha * np.linalg.norm(theta - mu, ord=1, axis=-1))
    
    def density_laplacian_mixture(self, theta): 
        den = [self.omegas[i] * self.multivariate_laplacian(theta, self.mus[i], self.alphas[i]) for i in range(len(self.alphas))]
        return sum(den)
    
    def potential_laplacian_mixture(self, theta): 
        return -np.log(self.density_laplacian_mixture(theta))
    
    def prox_uncentered_laplace(self, theta, gamma, mu):
        return mu + prox_laplace(theta - mu, gamma)
    
    def moreau_env_uncentered_laplace(self, theta, mu, alpha):
        prox = self.prox_uncentered_laplace(theta, self.lamda * alpha, mu)
        return alpha * np.linalg.norm(prox - mu, ord=1, axis=-1) + np.linalg.norm(prox - theta, ord=2, axis=-1)**2 / (2 * self.lamda)
    
    def smooth_multivariate_laplacian(self, theta, mu, alpha):
        return (alpha/2)**self.d * np.exp(-self.moreau_env_uncentered_laplace(theta, mu, alpha))
    
    def smooth_density_laplacian_mixture(self, theta): 
        den = [self.omegas[i] * self.smooth_multivariate_laplacian(theta, self.mus[i], self.alphas[i]) for i in range(len(self.alphas))]
        return sum(den)
    
    def smooth_potential_laplacian_mixture(self, theta): 
        return -np.log(self.smooth_density_laplacian_mixture(theta))

    def grad_smooth_density_multivariate_laplacian(self, theta, mu, alpha):        
        return self.smooth_multivariate_laplacian(theta, mu, alpha) * (self.prox_uncentered_laplace(theta, self.lamda * alpha, mu) - theta) / self.lamda
    
    def grad_smooth_density_laplacian_mixture(self, theta):
        grad_den = [self.omegas[i] * self.grad_smooth_density_multivariate_laplacian(theta, self.mus[i], self.alphas[i]) for i in range(len(self.alphas))]
        return sum(grad_den)
    
    def grad_smooth_potential_laplacian_mixture(self, theta):
        return -self.grad_smooth_density_laplacian_mixture(theta) / self.smooth_density_laplacian_mixture(theta)
    
    def hess_smooth_density_multivariate_laplacian(self, theta, mu, alpha):
        grad_laplace = (theta - self.prox_uncentered_laplace(theta, self.lamda * alpha, mu)) / self.lamda
        return (alpha/2)**self.d * np.exp(-alpha * np.linalg.norm(theta, ord=1, axis=-1)) * ( - np.eye(self.d) + np.outer(grad_laplace, grad_laplace))

    def hess_smooth_density_laplacian_mixture(self, theta):
        hess_den = [self.omegas[i] * self.hess_smooth_density_multivariate_laplacian(theta, self.mus[i], self.alphas[i]) for i in range(len(self.alphas))]
        return sum(hess_den)
        
    def hess_smooth_potential_laplacian_mixture(self, theta):
        density = self.smooth_density_laplacian_mixture(theta)
        grad_density = self.grad_smooth_density_laplacian_mixture(theta)
        hess_density = self.hess_smooth_density_laplacian_mixture(theta)
        return np.outer(grad_density, grad_density) / density**2 - hess_density / density
    
    def gd_update(self, theta, gamma): 
        return theta - gamma * self.grad_smooth_potential_laplacian_mixture(theta) 


    ## Unadjusted Langevin Algorithm (ULA)
    def ula(self, gamma):
        print("\nSampling with ULA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        # gammas = [gamma * i ** (-0.5) for i in range(1, self.n+1)]
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            theta_new = self.gd_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Metropolis-Adjusted Langevin Algorithm (MALA)
    def q_prob(self, theta1, theta2, gamma):
        return multivariate_normal(mean=self.gd_update(theta2, gamma), cov=2*gamma).pdf(theta1)

    def prob(self, theta_new, theta_old, gamma):
        density_ratio = self.smooth_density_laplacian_mixture(theta_new) / self.smooth_density_laplacian_mixture(theta_old)
        q_ratio = self.q_prob(theta_old, theta_new, gamma) / self.q_prob(theta_new, theta_old, gamma)
        return density_ratio * q_ratio

    def mala(self, gamma):
        print("\nSampling with MALA:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        # gammas = [gamma * i ** (-0.5) for i in range(1, self.n+1)]
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            theta_new = self.gd_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            p = self.prob(theta_new, theta0, gamma)
            alpha = min(1, p)
            if rng.random() <= alpha:
                theta.append(theta_new)    
                theta0 = theta_new
        return np.array(theta), len(theta)


    ## Preconditioned ULA 
    def preconditioned_gd_update(self, theta, gamma, M): 
        return theta - gamma * M @ self.grad_smooth_potential_laplacian_mixture(theta)

    def pula(self, gamma, M):
        print("\nSampling with Preconditioned Langevin Algorithm:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        # gammas = [gamma * i ** (-0.5) for i in range(1, self.n+1)]
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            theta_new = self.preconditioned_gd_update(theta0, gamma, M) + np.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    # Preconditioning with inverse Hessian
    def ihpula(self, gamma):
        print("\nSampling with Inverse Hessian Preconditioned Unadjusted Langevin Algorithm:")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        # gammas = [gamma * i ** (-0.5) for i in range(1, self.n+1)]
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            hess = self.hess_smooth_potential_laplacian_mixture(theta0)
            if len(self.mus) > 1:
                e = np.linalg.eigvals(hess)
                M = hess + (np.abs(min(e)) + .02) * np.eye(self.d)
                M = np.linalg.inv(M)
            else:
                M = np.linalg.inv(hess)
            theta_new = self.preconditioned_gd_update(theta0, gamma, M) + np.sqrt(2*gamma) * sqrtm(M) @ xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Mirror-Langevin Algorithm (MLA)
    def grad_mirror_hyp(self, theta, beta): 
        return np.arcsinh(theta / beta)

    def grad_conjugate_mirror_hyp(self, theta, beta):
        return beta * np.sinh(theta)

    def mla(self, gamma, beta):
        print("\nSampling with MLA: ")
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        # gammas = [gamma * i ** (-0.5) for i in range(1, self.n+1)]
        gammas = gamma * np.ones(self.n)
        for i in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            gamma = gammas[i]
            theta_new = self.grad_mirror_hyp(theta0, beta) - gamma * self.grad_smooth_potential_laplacian_mixture(theta0) + np.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
            theta_new = self.grad_conjugate_mirror_hyp(theta_new, beta)
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    # Cyclical step sizes
    def cyclical_gd_update(self, theta, gamma): 
        return theta - gamma * self.grad_smooth_potential_laplacian_mixture(theta)

    def cyclical_ula(self, gamma):
        rng = default_rng(self.seed)
        theta0 = rng.normal(0, 1, self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = self.cyclical_gd_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    '''
    def error(self, thetas1, thetas2):
        density_2d_gaussian_mixture(theta, mus, Sigmas)
        np.sum()

        return 
    '''


## Main function
def lmc_laplacian_mixture(gamma_ula=5e-2, gamma_mala=5e-2, 
                         gamma_pula=5e-2, gamma_mla=5e-2, 
                         lamda=1e-1, alpha=1e-1, n=5, K=5000, seed=0):
    # Our 2-dimensional distribution will be over variables X and Y
    N = 300
    X = np.linspace(-5, 5, N)
    Y = np.linspace(-5, 5, N)
    X, Y = np.meshgrid(X, Y)

    # location parameters
    mu1 = np.array([0., 0.])
    mu2 = np.array([-2., 3.])
    mu3 = np.array([2., -3.])
    mu4 = np.array([3., 3.])
    mu5 = np.array([-2., -2.])

    if n == 1:
        mus = [mu1]
    elif n == 2: 
        mus = [mu1, mu2]
    elif n == 3: 
        mus = [mu1, mu2, mu3]
    elif n == 4: 
        mus = [mu2, mu3, mu4, mu5]
    elif n == 5: 
        mus = [mu1, mu2, mu3, mu4, mu5]

    # scale parameters
    # alphas = np.arange(1, n + 1) * alpha
    alphas = np.ones(n) * alpha

    # weight vector
    omegas = np.ones(n) / n


    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y


    lmc_laplacian = LangevinMonteCarloLaplacian(mus, alphas, omegas, lamda, K, seed)

    # The distribution on the variables X, Y packed into pos.
    Z = lmc_laplacian.density_laplacian_mixture(pos)
    Z_smooth = lmc_laplacian.smooth_density_laplacian_mixture(pos)


    ## Plot of the true Laplacian mixture
    print("\nPlotting the true Laplacian mixture...")
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

    # plt.suptitle("True 2D Laplacian Mixture") 
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_{K}_1.pdf', dpi=500)


    print("\nPlotting the smoothed Laplacian mixture...")    
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    ax1.plot_surface(X, Y, Z_smooth, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax1.view_init(45, -70)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.contourf(X, Y, Z_smooth, zdir='z', offset=0, cmap=cm.viridis)
    ax2.view_init(90, 270)

    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    # plt.suptitle("2D Smoothed Laplacian Mixture") 
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_{K}_1_smooth.pdf', dpi=500)

    Z2 = lmc_laplacian.ula(gamma_ula)

    Z3, eff_K = lmc_laplacian.mala(gamma_mala)
    print(f'\nMALA acceptance rate: {eff_K / K} ')
        
    M = np.array([[1.0, 0.1], [0.1, 0.5]])
    Z4 = lmc_laplacian.pula(gamma_pula, M)
        
    # Z5 = lmc_laplacian.ihpula(gamma_ihpula)
      
    # beta = np.array([0.2, 0.8])
    beta = np.array([0.7, 0.3])
    Z6 = lmc_laplacian.mla(gamma_mla, beta)


    ## Plot of the true Laplacian mixture with 2d histograms of samples
    print("\nConstructing the 2D histograms of samples...")
    fig3, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].contourf(X, Y, Z_smooth, cmap=cm.viridis)
    axes[0,1].set_title("Smoothed density", fontsize=16)

    axes[0,2].hist2d(Z2[:,0], Z2[:,1], bins=100, cmap=cm.viridis)
    axes[0,2].set_title("ULA", fontsize=16)

    axes[1,0].hist2d(Z3[:,0], Z3[:,1], bins=100, cmap=cm.viridis)
    axes[1,0].set_title("MALA", fontsize=16)

    axes[1,1].hist2d(Z4[:,0], Z4[:,1], bins=100, cmap=cm.viridis)
    axes[1,1].set_title("PULA", fontsize=16)

    # axes[1,1].hist2d(Z5[:,0], Z5[:,1], bins=100, cmap=cm.viridis)
    # axes[1,1].set_title("IHPULA", fontsize=16)

    axes[1,2].hist2d(Z6[:,0], Z6[:,1], bins=100, cmap=cm.viridis)
    axes[1,2].set_title("MLA", fontsize=16)

    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig3.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_{K}_3.pdf', dpi=500)  


    ## Plot of the true Laplacian mixture with KDE of samples
    print("\nConstructing the KDEs of samples...")
    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    sns.set(font='serif', rc={'figure.figsize':(3.25, 3.5)})

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].contourf(X, Y, Z_smooth, cmap=cm.viridis)
    axes[0,1].set_title("Smoothed density", fontsize=16)

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("ULA", fontsize=16)

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("MALA", fontsize=16)

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("PULA", fontsize=16)

    # sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    # axes[1,1].set_title("IHPULA", fontsize=16)

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("MLA", fontsize=16)

    plt.show(block=False)
    plt.pause(10)
    plt.close()
    fig2.savefig(f'./fig/fig_laplace_n{n}_gamma{gamma_ula}_{K}_2.pdf', dpi=500)  


if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(lmc_laplacian_mixture)