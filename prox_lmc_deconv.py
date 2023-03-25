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
    
    def make_kernel_2D(self, PSF, dims):
        """
            PSF is the 2D kernel
            dims are is the side size of the image in order (r,c) 
        """
        d = len(PSF) ## assmuming square PSF (but not necessarily square image)
        print("kernel dimensions=", dims)
        N = dims[0]*dims[1]

        ## pre-fill a 2D matrix for the diagonals
        diags = np.zeros((d*d, N))
        offsets = np.zeros(d*d)
        heads = np.zeros(d*d) ## for this a list is OK
        i = 0
        for y in range(len(PSF)):
            for x in range(len(PSF[y])):
                diags[i,:] += PSF[y,x]
                heads[i] = PSF[y,x]
                xdist = d/2 - x 
                ydist = d/2 - y ## y direction pointing down
                offsets[i] = (ydist * dims[1] + xdist)
                i += 1
        ## create linear operator
        H = scipy.sparse.dia_matrix((diags,offsets), shape=(N,N))
        return H
    
    def make_blur_matrix(self, img, kernel_size=5):
        n = kernel_size
        k2 = np.zeros(shape=(n,n))
        k2[n/2,n/2] = 1
        sigma = kernel_size/5.0 ## 2.5 sigma
        testk = ndimage.uniform_filter(k2, sigma)  ## uniform filter
        blurmat = self.make_kernel_2D(testk, img.shape)
        return(blurmat)

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
    def PDHG(self, y, H, gamma0, gamma1, theta):
        print("\nOptimizing with Chambolle-Pock (PDHG):")
        M, N = y.shape
        rng = default_rng(self.seed)
        x0 = rng.standard_normal((M, N))
        u0 = tu0 = rng.standard_normal((M, N))
        x = []
        for _ in progress_bar(range(self.n)):
            x_new = prox.prox_square_loss(x0 - gamma0 * H.adjoint() * tu0, y, H, gamma0)
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

    g = ski.io.imread("fig/einstein.png")/255.0    
    M, N = g.shape    
    rng = default_rng(seed)
    # y5 = cv.boxFilter(g, ddepth=-1, ksize=(5, 5), normalize=False) + rng.standard_normal(size=(M, N)) * sigma
    # y5 = cv.blur(g, (5, 5)) + rng.standard_normal(size=(M, N)) * sigma
    # y5 = ndimage.uniform_filter(g, 5) + rng.standard_normal(size=(M, N)) * sigma
    # y6 = cv.boxFilter(g, ddepth=-1, ksize=(6, 6), normalize=False) + rng.standard_normal(size=(M, N)) * sigma
    # y6 = cv.blur(g, (6, 6)) + rng.standard_normal(size=(M, N)) * sigma    
    # y6 = ndimage.uniform_filter(g, 6) + rng.standard_normal(size=(M, N)) * sigma
    # y7 = cv.boxFilter(g, ddepth=-1, ksize=(7, 7), normalize=False) + rng.standard_normal(size=(M, N)) * sigma
    # y7 = cv.blur(g, (7, 7)) + rng.standard_normal(size=(M, N)) * sigma
    # y7 = ndimage.uniform_filter(g, 7) + rng.standard_normal(size=(M, N)) * sigma

    h5 = np.ones((5, 5))
    h5 /= h5.sum()
    nh5 = h5.shape
    H5 = pylops.signalprocessing.Convolve2D((M, N), h=h5, offset=(nh5[0] // 2, nh5[1] // 2))
    y5 = H5 * g + rng.standard_normal(size=(M, N)) * sigma

    h6 = np.ones((6, 6))
    h6 /= h6.sum()
    nh6 = h6.shape
    H6 = pylops.signalprocessing.Convolve2D((M, N), h=h6, offset=(nh6[0] // 2, nh6[1] // 2))
    y6 = H6 * g + rng.standard_normal(size=(M, N)) * sigma

    h7 = np.ones((7, 7))
    h7 /= h7.sum()
    nh7 = h7.shape
    H7 = pylops.signalprocessing.Convolve2D((M, N), h=h7, offset=(nh7[0] // 2, nh7[1] // 2))
    y7 = H7 * g + rng.standard_normal(size=(M, N)) * sigma

    # print(np.linalg.norm(ndimage.uniform_filter(g, 5) - H5 * g))
    # print(np.linalg.norm(ndimage.uniform_filter(g, 6) - H6 * g))
    # print(np.linalg.norm(ndimage.uniform_filter(g, 7) - H7 * g))

    # Plot of the original image and the blurred image
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.gray()  # show the filtered result in grayscale
    axes[0,0].imshow(g)
    axes[0,1].imshow(y5)
    axes[1,0].imshow(y6)
    axes[1,1].imshow(y7)
    # axes[1,0].imshow(y7)
    # axes[1,1].imshow(H5.adjoint() * H5 * g)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    # plt.show()

    prox_lmc = ProximalLangevinMonteCarloDeconvolution(lamda, sigma, tau, K, seed)  
    
    # x1 = prox_lmc.pgld(y5, H5, gamma_pgld)

    # x2 = prox_lmc.myula(y5, H5, gamma_myula)

    # x3, eff_K = prox_lmc.mymala(y5, H5, gamma_mymala)
    # print(f'\nMYMALA percentage of effective samples: {eff_K / K}')

    
    # tau = .5
    # x4 = prox_lmc.ulpda(y5, H5, gamma0_ulpda, gamma1_ulpda, theta, prox.prox_gaussian, prox.prox_tv)
    theta = 0.5
    x1 = prox_lmc.PDHG(y5, H5, gamma0_ulpda, gamma1_ulpda, theta)

    fig2, axes = plt.subplots(1, 3, figsize=(12, 8))
    plt.gray()  # show the filtered result in grayscale
    axes[0,0].imshow(g)
    axes[0,1].imshow(y5)
    axes[0,2].imshow(x1)

    plt.show(block=False)
    plt.pause(10)
    plt.close()

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