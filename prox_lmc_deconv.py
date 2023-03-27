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

import scipy
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from scipy import ndimage
import skimage as ski
from skimage import data, io, filters
import pylops
import cv2 as cv

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
    

    def posterior(self, x, y, H):   
        U = np.linalg.norm(y - H * x)**2 / (2*self.sigma**2)
        U += self.tau * self.total_variation(x)[0] * x
        U += self.tau * self.total_variation(x)[1] * x
        return np.exp(-U)


    def gd_update(self, x, y, H, gamma): 
        return x - gamma * H.adjoint() * (H * x - y) / (2*self.sigma**2)


    ## Proximal Gradient Langevin Dynamics (PGLD)
    def pgld(self, y, H, gamma):
        print("\nSampling with Proximal ULA:")
        d = y.shape[0]
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(d)
        theta = []
        for _ in progress_bar(range(self.n)):        
            xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
            theta0 = prox.prox_tv(theta0, self.lamda * self.tau)
            theta_new = self.gd_update(theta0, y, H, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)
    

    ## Moreau--Yosida Unadjusted Langevin Algorithm (MYULA)
    def grad_Moreau_env(self, theta):
        return (theta - prox.prox_tv(theta, self.lamda * self.tau)) / self.lamda

    def prox_update(self, theta, gamma):
        return -gamma * self.grad_Moreau_env(theta)

    def myula(self, y, H, gamma):
        print("\nSampling with MYULA:")
        d = y.shape[0]
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
            theta_new = self.gd_update(theta0, y, H, gamma) + self.prox_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            theta.append(theta_new)    
            theta0 = theta_new
        return np.array(theta)


    ## Moreau--Yosida regularized Metropolis-Adjusted Langevin Algorithm (MYMALA)
    def q_prob(self, theta1, theta2, gamma):
        return multivariate_normal(mean=self.gd_update(theta2, gamma) + self.prox_update(theta2, gamma), cov=2*gamma).pdf(theta1)


    def prob(self, theta_new, theta_old, y, H, gamma):
        density_ratio = self.posterior(theta_new, y, H) / self.posterior(theta_old, y, H)
        q_ratio = self.q_prob(theta_old, theta_new, gamma) / self.q_prob(theta_new, theta_old, gamma)
        return density_ratio * q_ratio


    def mymala(self, y, H, gamma):
        print("\nSampling with MYMALA:")
        d = y.shape[0]
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
            theta_new = self.gd_update(theta0, y, H, gamma) + self.prox_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            p = self.prob(theta_new, theta0, gamma)
            alpha = min(1, p)
            if rng.random() <= alpha:
                theta.append(theta_new) 
                theta0 = theta_new
        return np.array(theta), len(theta)


    # Unadjusted Langevin Primal-Dual Algorithm (ULPDA)
    def ulpda(self, y, H, gamma0, gamma1, theta, prox_f, prox_g):
        print("\nSampling with Unadjusted Langevin Primal-Dual Algorithm (ULPDA):")
        d = y.shape[0]
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(d)
        u0 = tu0 = rng.standard_normal(d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(d), np.eye(d))
            theta_new = prox_f(theta0 - gamma0 * H.adjoint() * tu0, gamma0) + np.sqrt(2*gamma0) * xi
            u_new = prox.prox_conjugate(u0 + gamma1 * H * (2*theta_new - theta0), gamma1, prox_g)
            tu_new = u0 + theta * (u_new - u0)
            theta.append(theta_new)    
            theta0 = theta_new
            u0 = u_new
            tu0 = tu_new
        return np.array(theta)
    
    # PDHG (Chambolle--Pock)
    def PDHG(self, y, h, H, gamma0, gamma1, theta):
        print("\nOptimizing with Chambolle-Pock (PDHG):")
        M, N = y.shape
        rng = default_rng(self.seed)
        x0 = rng.standard_normal((M, N))
        u0 = tu0 = rng.standard_normal((M, N))
        x = []
        for _ in progress_bar(range(self.n)):
            # x_new = prox.prox_square_loss(x0 - gamma0 * H.adjoint() * tu0, y, H, gamma0)
            x_new = pyproximal.L2Convolve(h, b=y, sigma=gamma0)(x0 - gamma0 * H.adjoint() * tu0)
            u_new = prox.prox_conjugate(u0 + gamma1 * H * (2*x_new - x0), gamma1, prox.prox_laplace)
            theta = 1 / np.sqrt(1 + 2 * 0.1 * gamma0)
            gamma0 *= theta
            gamma1 /= theta
            tu_new = u0 + theta * (u_new - u0)
            # x.append(x_new)
            x0 = x_new
            u0 = u_new
            tu0 = tu_new
        # return np.array(x)
        return x_new

## Main function
def prox_lmc_deconv(gamma_pgld=5e-2, gamma_myula=5e-2, 
                    gamma_mymala=5e-2, gamma0_ulpda=5e-2, gamma1_ulpda=5e-2, 
                    lamda=0.01, sigma=0.47, tau=0.03, K=10000, seed=0):

    # img = data.camera()
    img = io.imread("fig/einstein.png")
    ny, nx = img.shape
    rng = default_rng(seed)
    # y5 = cv.boxFilter(img, ddepth=-1, ksize=(5, 5), normalize=False) + rng.normal(0, sigma, size=(ny, nx))
    # y5 = cv.blur(img, (5, 5)) + rng.normal(0, sigma, size=(ny, nx))
    # y5 = ndimage.uniform_filter(img, 5) + rng.normal(0, sigma, size=(ny, nx))
    # y6 = cv.boxFilter(img, ddepth=-1, ksize=(6, 6), normalize=False) + rng.normal(0, sigma, size=(ny, nx))
    # y6 = cv.blur(img, (6, 6)) + rng.normal(0, sigma, size=(ny, nx))
    # y6 = ndimage.uniform_filter(img, 6) + rng.normal(0, sigma, size=(ny, nx))
    # y7 = cv.boxFilter(img, ddepth=-1, ksize=(7, 7), normalize=False) + rng.normal(0, sigma, size=(ny, nx))
    # y7 = cv.blur(img, (7, 7)) + rng.normal(0, sigma, size=(ny, nx))
    # y7 = ndimage.uniform_filter(img, 7) + rng.normal(0, sigma, size=(ny, nx))

    h5 = np.ones((5, 5))
    h5 /= h5.sum()
    nh5 = h5.shape
    H5 = pylops.signalprocessing.Convolve2D((ny, nx), h=h5, offset=(nh5[0] // 2, nh5[1] // 2))
    y5 = H5 * img + rng.normal(0, sigma, size=(ny, nx))
    # y5 = img + rng.normal(0, sigma, size=(ny, nx))

    h6 = np.ones((6, 6))
    h6 /= h6.sum()
    nh6 = h6.shape
    H6 = pylops.signalprocessing.Convolve2D((ny, nx), h=h6, offset=(nh6[0] // 2, nh6[1] // 2))
    y6 = H6 * img + rng.normal(0, sigma, size=(ny, nx))

    h7 = np.ones((7, 7))
    h7 /= h7.sum()
    nh7 = h7.shape
    H7 = pylops.signalprocessing.Convolve2D((ny, nx), h=h7, offset=(nh7[0] // 2, nh7[1] // 2))
    y7 = H7 * img + rng.normal(0, sigma, size=(ny, nx))


    # print(np.linalg.norm(ndimage.uniform_filter(g, 5) - H5 * g))
    # print(np.linalg.norm(ndimage.uniform_filter(g, 6) - H6 * g))
    # print(np.linalg.norm(ndimage.uniform_filter(g, 7) - H7 * g))

    # Plot of the original image and the blurred image
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.gray()  # show the filtered result in grayscale
    axes[0,0].imshow(img)
    axes[0,1].imshow(y5)
    axes[1,0].imshow(y6)
    axes[1,1].imshow(y7)
    # axes[1,0].imshow(y7)
    # axes[1,1].imshow(H5.adjoint() * H5 * g)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    # plt.show()


    # Gradient operator
    sampling = 1.
    Gop = pylops.Gradient(dims=(ny, nx), sampling=sampling, edge=False, kind='forward', dtype='float64')
    L = 8. / sampling ** 2 # maxeig(Gop^H Gop)

    # L2 data term
    l2 = pyproximal.L2(Op=H7, b=y7.ravel(), niter=50, warm=True)

    # L1 regularization (isotropic TV)
    l1iso = pyproximal.L21(ndim=2, sigma=tau)

    # Primal-dual
    def callback(x, f, g, K, cost, xtrue, err):
        cost.append(f(x) + g(K.matvec(x)))
        err.append(np.linalg.norm(x - xtrue))

    tau0 = 0.95 / np.sqrt(L)
    mu0 = 0.95 / np.sqrt(L)

    cost_fixed = []
    err_fixed = []
    iml12_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=K, show=True,
                                                    callback=lambda x: callback(x, l2, l1iso,
                                                                                Gop, cost_fixed,
                                                                                img.ravel(),
                                                                                err_fixed))
    iml12_fixed = iml12_fixed.reshape(img.shape)

    cost_ada = []
    err_ada = []
    iml12_ada, steps = \
        pyproximal.optimization.primaldual.AdaptivePrimalDual(l2, l1iso, Gop,
                                                            tau=tau0, mu=mu0,
                                                            x0=np.zeros_like(img.ravel()),
                                                            niter=K//2, show=True, tol=0.05,
                                                            callback=lambda x: callback(x, l2, l1iso,
                                                                                        Gop, cost_ada,
                                                                                        img.ravel(),
                                                                                        err_ada))
    iml12_ada = iml12_ada.reshape(img.shape)

    print(np.linalg.norm(iml12_fixed - img))
    print(np.linalg.norm(iml12_ada - img))
    

    prox_lmc = ProximalLangevinMonteCarloDeconvolution(lamda, sigma, tau, K, seed)  
    
    # x1 = prox_lmc.pgld(y5, H5, gamma_pgld)

    # x2 = prox_lmc.myula(y5, H5, gamma_myula)

    # x3, eff_K = prox_lmc.mymala(y5, H5, gamma_mymala)
    # print(f'\nMYMALA percentage of effective samples: {eff_K / K}')

    
    # tau = .5
    # x4 = prox_lmc.ulpda(y5, H5, gamma0_ulpda, gamma1_ulpda, theta, prox.prox_gaussian, prox.prox_tv)
    # theta = 0.5
    # x1 = prox_lmc.PDHG(y5, h5, H5, gamma0_ulpda, gamma1_ulpda, theta)

    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.gray()  # show the filtered result in grayscale
    axes[0,0].imshow(img)
    axes[0,1].imshow(y5)
    axes[1,0].imshow(iml12_fixed)
    axes[1,1].imshow(iml12_ada)

    plt.show()
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()

    '''
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