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
Usage: python prox_lmc_deconv.py --gamma_pgld=5e-2 --gamma_myula=5e-2 --gamma_mymala=5e-2 --gamma_fbula=5e-2 --gamma0_ulpda=5e-2 --gamma1_ulpda=5e-2 --alpha=1.5e-1 --lamda=2.5e-1 --t=100 --seed=0 --K=10000 --n=5
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
from scipy import ndimage
import skimage as ski
from skimage import data, io, filters
import pylops

import prox
# import prox_tv as ptv
import pyproximal


class ProximalLangevinMonteCarloDeconvolution:
    def __init__(self, lamda, sigma, tau, K=10000, seed=0) -> None:
        super(ProximalLangevinMonteCarloDeconvolution, self).__init__()
        self.lamda = lamda
        self.sigma = sigma
        self.tau = tau
        self.n = K
        self.seed = seed

    def total_variation(self, g):
        nx, ny = g.shape
        tv_x = pylops.FirstDerivative((ny, nx), axis=0, edge=False, kind="backward")
        tv_y = pylops.FirstDerivative((ny, nx), axis=1, edge=False, kind="backward")
        return tv_x, tv_y


    def posterior(self, x, y, size=5):   
        U = np.linalg.norm(y - ndimage.uniform_filter(x, size=size))**2 / (2*self.sigma**2)
        U += self.tau * self.total_variation(x)[0] * x
        U += self.tau * self.total_variation(x)[1] * x
        return np.exp(-U)


    def gd_update(self, x, y, H, gamma): 
        return x - gamma * H @ (H @ x - y) / (2*self.sigma**2)


    ## Proximal Gradient Langevin Dynamics (PGLD)
    def pgld(self, gamma):
        print("\nSampling with Proximal ULA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):        
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta0 = prox.prox_laplace(theta0, self.lamda * self.alpha)
            theta_new = self.gd_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)
    

    ## Moreau--Yosida Unadjusted Langevin Algorithm (MYULA)
    def grad_Moreau_env(self, theta):
        return (theta - prox.prox_laplace(theta, self.lamda * self.alpha)) / self.lamda

    def prox_update(self, theta, gamma):
        return -gamma * self.grad_Moreau_env(theta)

    def myula(self, gamma):
        print("\nSampling with MYULA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
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
        density_ratio = ((self.density_gaussian_mixture(theta_new) * self.multivariate_laplacian(theta_new)) / 
                        (self.density_gaussian_mixture(theta_old) * self.multivariate_laplacian(theta_old)))
        q_ratio = self.q_prob(theta_old, theta_new, gamma) / self.q_prob(theta_new, theta_old, gamma)
        return density_ratio * q_ratio


    def mymala(self, gamma):
        print("\nSampling with MYMALA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
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


    ## Forward-Backward Unadjusted Langevin Algorithm (FBULA)
    def grad_FB_env(self, theta):
        return (np.eye(theta.shape[0]) - self.lamda * self.hess_potential_gaussian_mixture(theta)) @ (theta - prox.prox_laplace(self.gd_update(theta, self.lamda), self.lamda * self.alpha)) / self.lamda

    def gd_FB_update(self, theta, gamma):
        return theta - gamma * self.grad_FB_env(theta)

    def fbula(self, gamma):
        print("\nSampling with FBULA:")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = self.gd_FB_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)



    # Unadjusted Langevin Primal-Dual Algorithm (ULPDA)
    def ulpda(self, gamma0, gamma1, tau, D, prox_f, prox_g):
        print("\nSampling with Unadjusted Langevin Primal-Dual Algorithm (ULPDA):")
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(self.d)
        u0 = tu0 = rng.standard_normal(self.d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d))
            theta_new = prox_f(theta0 - gamma0 * D.T @ tu0, gamma0) + np.sqrt(2*gamma0) * xi
            u_new = prox.prox_conjugate(u0 + gamma1 * D @ (2*theta_new - theta0), gamma1, prox_g)
            tu_new = u0 + tau * (u_new - u0)
            theta.append(theta_new)    
            theta0 = theta_new
            u0 = u_new
            tu0 = tu_new
        return np.array(theta)


## Main function
def prox_lmc_deconv(gamma_pgld=5e-2, gamma_myula=5e-2, 
                    gamma_mymala=5e-2, gamma_fbula=5e-2, 
                    gamma0_ulpda=5e-2, gamma1_ulpda=5e-2, 
                    lamda=0.01, sigma=0.47, tau=0.03, K=10000, seed=0):

    g = ski.io.imread("fig/einstein.png")/255.0    
    M, N = g.shape    
    rng = default_rng(seed)
    noisy_g = ndimage.uniform_filter(g, size=5) + rng.standard_normal(size=(M, N)) * sigma

    # Plot of the original image and the blurred image
    fig = plt.figure(figsize=(13, 4))
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(g)
    ax2.imshow(noisy_g)
    plt.show(block=False)
    plt.pause(10)
    plt.close()


    prox_lmc = ProximalLangevinMonteCarloDeconvolution(lamda, sigma, tau, K, seed)  
    




    '''
    Z1 = prox_lmc.pgld(gamma_pgld)

    Z2 = prox_lmc.myula(gamma_myula)

    Z3, eff_K = prox_lmc.mymala(gamma_mymala)
    print(f'\nMYMALA percentage of effective samples: {eff_K / K}')


    Z4 = prox_lmc.fbula(gamma_fbula)
    

    D = np.eye(2)
    tau = .5
    Z5 = prox_lmc.ulpda(gamma0_ulpda, gamma1_ulpda, tau, D, prox.prox_gaussian, prox.prox_laplace)
    

    ## Plot of the true Gaussian mixture with 2d histograms of samples
    print("\nConstructing the 2D histograms of samples...")
    ran = [[xmin, xmax], [ymin, ymax]]
    fig3, axes = plt.subplots(2, 4, figsize=(17, 8))

    # axes[0,0].hist2d(Z[:, 0], Z[:, 1], bins=100, density=True)
    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].contourf(X, Y, Z_smooth, cmap=cm.viridis)
    axes[0,1].set_title("Smoothed density", fontsize=16)

    axes[0,2].hist2d(Z1[:,0], Z1[:,1], bins=100, range=ran, cmap=cm.viridis)
    # sns.kdeplot(x=Z1[:,0], y=Z1[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,2].set_title("PGLD", fontsize=16)

    axes[0,3].hist2d(Z2[:,0], Z2[:,1], bins=100, range=ran, cmap=cm.viridis)
    # sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,3].set_title("MYULA", fontsize=16)

    axes[1,0].hist2d(Z3[:,0], Z3[:,1], bins=100, range=ran, cmap=cm.viridis)
    # sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,3])
    axes[1,0].set_title("MYMALA", fontsize=16)

    axes[1,1].hist2d(Z4[:,0], Z4[:,1], bins=100, range=ran, cmap=cm.viridis)
    # sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,1].set_title("PP-ULA", fontsize=16)

    axes[1,2].hist2d(Z5[:,0], Z5[:,1], bins=100, range=ran, cmap=cm.viridis)
    # sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,2].set_title("FBULA", fontsize=16)

    axes[1,3].hist2d(Z6[:,0], Z6[:,1], bins=100, range=ran, cmap=cm.viridis)
    # sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,3].set_title("LBMUMLA", fontsize=16)

    # axes[1,3].hist2d(Z7[:,0], Z7[:,1], bins=100, range=ran, cmap=cm.viridis)
    # sns.kdeplot(x=Z7[:,0], y=Z7[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,3])
    # axes[1,3].set_title("ULPDA", fontsize=16)

    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig3.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_lambda{lamda}_{K}_3.pdf', dpi=500)

    
    ## Plot of the true Gaussian mixture with KDE of samples
    print("\nConstructing the KDEs of samples...")
    # fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig2, axes = plt.subplots(2, 4, figsize=(17, 8))
    # fig2.suptitle("True density and KDEs of samples") 

    sns.set(font='serif', rc={'figure.figsize':(3.25, 3.5)})

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    axes[0,1].contourf(X, Y, Z_smooth, cmap=cm.viridis)
    axes[0,1].set_title("Smoothed density", fontsize=16)

    sns.kdeplot(x=Z1[:,0], y=Z1[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("PGLD", fontsize=16)

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,3])
    axes[0,3].set_title("MYULA", fontsize=16)

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("MYMALA", fontsize=16)

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("PP-ULA", fontsize=16)

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("FBULA", fontsize=16)

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,3])
    axes[1,3].set_title("LBMUMLA", fontsize=16)

    # sns.kdeplot(x=Z7[:,0], y=Z7[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,3])
    # axes[1,3].set_title("ULPDA", fontsize=16)
    
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig2.savefig(f'./fig/fig_prox_n{n}_gamma{gamma_pgld}_lambda{lamda}_{K}_2.pdf', dpi=500)  
    '''

if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(prox_lmc_deconv)