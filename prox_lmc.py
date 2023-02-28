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

# Usage: python prox_lmc.py --gamma_proxula=7.5e-2 --gamma_myula=7.5e-2  --gamma_mymala=7.5e-2\
# --gamma_ppula=8e-2 --gamma_eula=5e-4 --gamma_bmumla=5e-2 --K=10000 --n=5

import os
import itertools
from fastprogress import progress_bar
from typing import NamedTuple
import fire

import random
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import seaborn as sns
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    } 
    )

from scipy.linalg import sqrtm
from scipy.stats import kde, multivariate_normal
from scipy.integrate import quad, dblquad

from prox import *


def multivariate_gaussian(theta, mu, Sigma):
    """Return the multivariate Gaussian distribution on array theta."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * np.abs(Sigma_det))
    # This einsum call calculates (theta - mu)T.Sigma-1.(theta - mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', theta - mu, Sigma_inv, theta - mu)

    return np.exp(-fac / 2) / N


def density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas): 
    K = len(mus)
    den = [lambdas[k] * multivariate_gaussian(theta, mus[k], Sigmas[k]) for k in range(K)]
    return sum(den)


def potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas): 
    return -np.log(density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas))


def prior(theta, alpha):    
    n = theta.shape[0]
    return (alpha/2)**n * np.exp(-alpha * np.linalg.norm(theta, axis=-1))


def grad_density_multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * np.abs(Sigma_det))
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)    
    return np.exp(-fac / 2) / N * Sigma_inv @ (mu - pos)


def grad_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas):
    K = len(mus)
    grad_den = [lambdas[k] * grad_density_multivariate_gaussian(theta, mus[k], Sigmas[k]) for k in range(K)]
    return sum(grad_den)


def grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas):
    return -grad_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas) / density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)


def hess_density_multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * np.abs(Sigma_det))
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N * (Sigma_inv @ np.outer(pos - mu, pos - mu) @ Sigma_inv - Sigma_inv)


def hess_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas):
    K = len(mus)
    hess_den = [lambdas[k] * hess_density_multivariate_gaussian(theta, mus[k], Sigmas[k]) for k in range(K)]
    return sum(hess_den)
    

def hess_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas):
    density = density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)
    grad_density = grad_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)
    hess_density = hess_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)
    return np.outer(grad_density, grad_density) / density**2 - hess_density / density


def gd_update(theta, mus, Sigmas, lambdas, gamma): 
    return theta - gamma * grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas) 


## Proximal Unadjusted Langevin Algorithm (P-ULA)
def prox_ula_gaussian_mixture(gamma, mus, Sigmas, lambdas, lamda, alpha, n=1000, seed=0):
    d = mus[0].shape[0]
    print("\nSampling with Proximal ULA:")
    rng = default_rng(seed)
    theta0 = rng.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):        
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta0 = prox_laplace(theta0, lamda * alpha)
        theta_new = gd_update(theta0, mus, Sigmas, lambdas, gamma) + np.sqrt(2*gamma) * xi
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)


## Moreau--Yosida Unadjusted Langevin Algorithm (MYULA)
def grad_Moreau_env(theta, lamda, alpha):
    return (theta - prox_laplace(theta, lamda * alpha)) / lamda

def prox_update(theta, gamma, lamda, alpha):
    return -gamma * grad_Moreau_env(theta, lamda, alpha)

def myula_gaussian_mixture(gamma, mus, Sigmas, lambdas, lamda, alpha, n=1000, seed=0):
    d = mus[0].shape[0]
    print("\nSampling with MYULA:")
    rng = default_rng(seed)
    theta0 = rng.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = gd_update(theta0, mus, Sigmas, lambdas, gamma) + prox_update(theta0, gamma, lamda, alpha) + np.sqrt(2*gamma) * xi
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)


## Moreau--Yosida regularized Metropolis-Adjusted Langevin Algorithm (MYMALA)
def q_prob(theta1, theta2, gamma, mus, Sigmas, lambdas, lamda, alpha):
    return multivariate_normal(mean=gd_update(theta2, mus, Sigmas, lambdas, gamma) + prox_update(theta2, gamma, lamda, alpha), cov=2*gamma).pdf(theta1)


def prob(theta_new, theta_old, gamma, mus, Sigmas, lambdas, lamda, alpha):
    density_ratio = ((density_2d_gaussian_mixture(theta_new, mus, Sigmas, lambdas) * prior(theta_new, alpha)) / 
                     (density_2d_gaussian_mixture(theta_old, mus, Sigmas, lambdas) * prior(theta_old, alpha)))
    q_ratio = q_prob(theta_old, theta_new, gamma, mus, Sigmas, lambdas, lamda, alpha) / q_prob(theta_new, theta_old, gamma, mus, Sigmas, lambdas, lamda, alpha)
    return density_ratio * q_ratio


def mymala_gaussian_mixture(gamma, mus, Sigmas, lambdas, lamda, alpha, n=1000, seed=0):
    d = mus[0].shape[0]
    print("\nSampling with MYMALA:")
    rng = default_rng(seed)
    theta0 = rng.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = gd_update(theta0, mus, Sigmas, lambdas, gamma) + prox_update(theta0, gamma, lamda, alpha) + np.sqrt(2*gamma) * xi
        p = prob(theta_new, theta0, gamma, mus, Sigmas, lambdas, lamda, alpha)
        alpha = min(1, p)
        if random.random() <= alpha:
            theta.append(theta_new)    
            theta0 = theta_new
    return np.array(theta), len(theta)


## Preconditioned Proximal ULA 
def preconditioned_gd_update(theta, mus, Sigmas, lambdas, gamma, M): 
    return theta - gamma * M @ grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)

def preconditioned_prox(x, gamma, M): 
    rho = 1 / np.linalg.norm(M, ord=2)
    eps = min(1, rho)
    eta = 2 * rho - eps
    w = np.zeros(x.shape[0])
    for _ in range(100):
        u = x - M @ w
        w += eta * u - eta * prox_laplace(eta*w + u, 1/eta * gamma)
    return w

def preconditioned_prox_update(theta, mus, Sigmas, lambdas, gamma, M):
    return preconditioned_prox(theta - gamma * M @ grad_density_2d_gaussian_mixture(theta, mus, Sigmas, lambdas), gamma, M)

def ppula_gaussian_mixture(gamma, mus, Sigmas, lambdas, M, n=1000, seed=0):
    d = mus[0].shape[0]
    print("\nSampling with PP-ULA:")
    rng = default_rng(seed)
    theta0 = rng.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = preconditioned_prox_update(theta0, mus, Sigmas, lambdas, gamma, M) + np.sqrt(2*gamma) * sqrtm(M) @ xi
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)


## Envelope Unadjusted Langevin Algorithm (EULA)
def grad_FB_env(theta, mus, Sigmas, lambdas, lamda, alpha):
    return (np.eye(theta.shape[0]) - lamda * hess_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)) @ (theta - prox_laplace(gd_update(theta, mus, Sigmas, lambdas, alpha), lamda * alpha)) / lamda

def gd_FB_update(theta, gamma, mus, Sigmas, lambdas, lamda, alpha):
    return theta - gamma * grad_FB_env(theta, mus, Sigmas, lambdas, lamda, alpha)

def eula_gaussian_mixture(gamma, mus, Sigmas, lambdas, lamda, alpha, n=1000, seed=0):
    d = mus[0].shape[0]
    print("\nSampling with EULA:")
    rng = default_rng(seed)
    theta0 = rng.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = gd_FB_update(theta0, gamma, mus, Sigmas, lambdas, lamda, alpha) + np.sqrt(2*gamma) * xi
        theta.append(theta_new)    
        theta0 = theta_new
    return np.array(theta)


## Bregman--Moreau Unadjusted Mirror-Langevin Algorithm (BMUMLA)
def grad_mirror_hyp(theta, beta): 
    return np.arcsinh(theta / beta)

def grad_conjugate_mirror_hyp(theta, beta):
    return beta * np.sinh(theta)

def left_bregman_prox_ell_one_hypent(theta, beta, gamma):
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

def grad_BM_env(theta, beta, lamda, alpha):
    return 1/lamda * (theta**2 + beta**2)**(-.5) * (theta - left_bregman_prox_ell_one_hypent(theta, beta, lamda * alpha))

def gd_BM_update(theta, gamma, mus, Sigmas, lambdas):
    return -gamma * grad_potential_2d_gaussian_mixture(theta, mus, Sigmas, lambdas)

def prox_BM_update(theta, beta, gamma, lamda, alpha):
    return -gamma * grad_BM_env(theta, beta, lamda, alpha)

def lbmumla_gaussian_mixture(gamma, mus, Sigmas, lambdas, beta, sigma, lamda, alpha, n=1000, seed=0):
    d = mus[0].shape[0]
    print("\nSampling with LBMUMLA: ")
    rng = default_rng(seed)
    theta0 = rng.normal(0, 1, d)
    theta = []
    for _ in progress_bar(range(n)):
        xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
        theta_new = grad_mirror_hyp(theta0, beta) + gd_BM_update(theta0, gamma, mus, Sigmas, lambdas) + prox_BM_update(theta0, sigma, gamma, lamda, alpha) + np.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
        theta_new = grad_conjugate_mirror_hyp(theta_new, beta)
        theta.append(theta_new) 
        theta_new = grad_mirror_hyp(theta0, beta) - gamma *  + np.sqrt(2*gamma) * (theta0**2 + beta**2)**(-.25) * xi
        theta0 = theta_new
    return np.array(theta)



'''
def error(theta, mus, Sigmas, lambdas):
    density_2d_gaussian_mixture(theta, mus, Sigmas)
    np.sum()

    return 
'''


def plot_hist2d(z, title): 
    z0 = z[:,0]
    z1 = z[:,1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    hist, xedges, yedges = np.histogram2d(z0, z1, bins=50, range=[[-5, 5], [-5, 5]], density=True)

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_title(title)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.show()


def plot_contour_hist2d(z, title, bins=50):
    counts, xbins, ybins, image = plt.hist2d(z[:,0], z[:,1], bins=bins, norm=LogNorm(), cmap=cm.viridis)
    plt.colorbar()
    plt.title(title)
    plt.show()


## Main function
def prox_lmc_gaussian_mixture(gamma_proxula=7.5e-2, gamma_myula=7.5e-2, 
                                gamma_mymala=7.5e-2, gamma_ppula=8e-2, 
                                gamma_eula=5e-4, gamma_lbmumla=5e-2, 
                                lamda=0.01, alpha=.1, n=2, K=10000):
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
    lambdas = np.ones(n) / n

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    
    Z = density_2d_gaussian_mixture(pos, mus, Sigmas, lambdas) * prior(pos, alpha)
    # Z = prior(pos, alpha)
    # Z = np.exp(np.log(density_2d_gaussian_mixture(pos, mus, Sigmas, lambdas)) - 0.005 * np.sum(np.abs(pos)))
    # print(Z.shape)

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
    # plt.show()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    fig.savefig(f'./fig/fig_prox_{n}_1.pdf', dpi=500)


    Z2 = prox_ula_gaussian_mixture(gamma_proxula, mus, Sigmas, lambdas, lamda, alpha, n=K)

    Z3 = myula_gaussian_mixture(gamma_myula, mus, Sigmas, lambdas, lamda, alpha, n=K)

    Z4, eff_K = mymala_gaussian_mixture(gamma_mymala, mus, Sigmas, lambdas, lamda, alpha, n=K)
    print(f'\nMYMALA acceptance rate: {eff_K / K} ')

    M = np.array([[1.0, 0.1], [0.1, 0.5]])
    Z5 = ppula_gaussian_mixture(gamma_ppula, mus, Sigmas, lambdas, M, n=K)

    Z6 = eula_gaussian_mixture(gamma_eula, mus, Sigmas, lambdas, lamda, alpha, n=K)
    
    # beta = np.array([0.2, 0.8])
    beta = np.array([0.7, 0.3])
    sigma = (alpha * np.ones(2))**2
    Z7 = lbmumla_gaussian_mixture(gamma_lbmumla, mus, Sigmas, lambdas, beta, sigma, lamda, alpha, n=K)
    print("\n")

    ## Plot of the true Gaussian mixture with KDE of samples
    print("Constructing the plots of samples...")
    # fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig2, axes = plt.subplots(2, 4, figsize=(17, 8))
    # fig2.suptitle("True density and KDEs of samples") 

    sns.set(rc={'figure.figsize':(3.25, 3.5)})

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density", fontsize=16)

    sns.kdeplot(x=Z2[:,0], y=Z2[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,1].set_title("Proximal ULA", fontsize=16)

    sns.kdeplot(x=Z3[:,0], y=Z3[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,2])
    axes[0,2].set_title("MYULA", fontsize=16)

    sns.kdeplot(x=Z4[:,0], y=Z4[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,3])
    axes[0,3].set_title("PP-ULA", fontsize=16)

    sns.kdeplot(x=Z5[:,0], y=Z5[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,0])
    axes[1,0].set_title("MYMALA", fontsize=16)

    sns.kdeplot(x=Z6[:,0], y=Z6[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,1])
    axes[1,1].set_title("EULA", fontsize=16)

    sns.kdeplot(x=Z7[:,0], y=Z7[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[1,2])
    axes[1,2].set_title("LBMUMLA", fontsize=16)

    plt.show()
    # plt.pause(5)
    # plt.close()
    fig2.savefig(f'./fig/fig_prox_{n}_2.pdf', dpi=500)  




if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(prox_lmc_gaussian_mixture)