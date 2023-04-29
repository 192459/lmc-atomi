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

# Install libraries: 
# pip install -U numpy matplotlib scipy seaborn fire fastprogress SciencePlots scikit-image pylops pyproximal arviz

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
    "text.usetex": True,
    } 
    )

import scipy
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from scipy import ndimage

import skimage as ski
from skimage import data, io, filters
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

import pylops
import pyproximal

import arviz as az

import prox


def signal_noise_ratio(image_true, image_test): 
    return 20 * np.log10(np.linalg.norm(image_true) / np.linalg.norm(image_test - image_true))

class ProximalLangevinMonteCarloDeconvolution:
    def __init__(self, lamda, sigma, tau, N=10000, seed=0) -> None:
        super(ProximalLangevinMonteCarloDeconvolution, self).__init__()
        self.lamda = lamda
        self.sigma = sigma
        self.tau = tau
        self.n = N
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


    ## Moreau--Yosida Unadjusted Langevin Algorithm (MYULA)
    def grad_Moreau_env(self, theta):
        return (theta - pyproximal.TV(dims=theta.shape, sigma=self.lamda * self.tau)(theta)) / self.lamda

    def prox_update(self, theta, gamma):
        return -gamma * self.grad_Moreau_env(theta)

    def myula(self, y, H, gamma):
        print("\nSampling with MYULA:")
        d = y.shape[0]
        rng = default_rng(self.seed)
        theta0 = rng.standard_normal(d)
        theta = []
        for _ in progress_bar(range(self.n)):
            xi = rng.multivariate_normal(np.zeros(d), np.identity(d))
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
            xi = rng.multivariate_normal(np.zeros(d), np.identity(d))
            theta_new = self.gd_update(theta0, y, H, gamma) + self.prox_update(theta0, gamma) + np.sqrt(2*gamma) * xi
            p = self.prob(theta_new, theta0, gamma)
            alpha = min(1, p)
            if rng.random() <= alpha:
                theta.append(theta_new) 
                theta0 = theta_new
        return np.array(theta), len(theta)


## Main function
def prox_lmc_deconv(gamma_myula=5e-2, gamma_ulpda=5e-1, lamda=0.01, sigma=0.75, tau=0.03, alpha=0.8,
                    N=10000, niter_l2=50, niter_tv=10, niter_map=1000, image='camera', alg='ULPDA', 
                    computeMAP=False, seed=0):

    # Choose the test image
    if image == 'einstein':
        img = io.imread("fig/einstein.png")
    elif image == 'camera':
        img = data.camera()
        
    ny, nx = img.shape
    rng = default_rng(seed)
    # snr = 20.
    # sigma = np.linalg.norm(img.ravel(), np.inf) * 10**(-snr/20)

    ###
    h5 = np.ones((5, 5))
    h5 /= h5.sum()
    nh5 = h5.shape
    H5 = pylops.signalprocessing.Convolve2D((ny, nx), h=h5, offset=(nh5[0] // 2, nh5[1] // 2))
    y = H5 * img + rng.normal(0, sigma, size=(ny, nx))

    h6 = np.ones((6, 6))
    h6 /= h6.sum()
    nh6 = h6.shape
    H6 = pylops.signalprocessing.Convolve2D((ny, nx), h=h6, offset=(nh6[0] // 2, nh6[1] // 2))

    h7 = np.ones((7, 7))
    h7 /= h7.sum()
    nh7 = h7.shape
    H7 = pylops.signalprocessing.Convolve2D((ny, nx), h=h7, offset=(nh7[0] // 2, nh7[1] // 2))


    # Plot of the original image and the blurred image
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    plt.gray()  # show the filtered result in grayscale
    axes[0].imshow(img)
    axes[1].imshow(y)
    plt.show(block=False)
    plt.pause(5)
    plt.close() 


    # Gradient operator
    sampling = 1.
    Gop = pylops.Gradient(dims=(ny, nx), sampling=sampling, edge=False, kind='forward', dtype='float64')
    
    # L2 data term
    l2_5 = pyproximal.L2(Op=H5, b=y.ravel(), sigma=1/sigma**2, niter=50, warm=True)
    l2_6 = pyproximal.L2(Op=H6, b=y.ravel(), sigma=1/sigma**2, niter=50, warm=True)
    l2_7 = pyproximal.L2(Op=H7, b=y.ravel(), sigma=1/sigma**2, niter=50, warm=True)

    # L2 data term - Moreau envelope of anisotropic TV
    l2_5_mc = prox.L2_ncvx_tv(dims=(ny, nx), Op=H5, Op2=Gop, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, niter=niter_l2, warm=True)
    l2_6_mc = prox.L2_ncvx_tv(dims=(ny, nx), Op=H6, Op2=Gop, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, niter=niter_l2, warm=True)
    l2_7_mc = prox.L2_ncvx_tv(dims=(ny, nx), Op=H7, Op2=Gop, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, niter=niter_l2, warm=True)

    # L2 data term - Moreau envelope of anisotropic TV
    l2_5_me_a = prox.L2_ncvx_tv(dims=(ny, nx), Op=H5, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, niter=niter_l2, warm=True)
    l2_6_me_a = prox.L2_ncvx_tv(dims=(ny, nx), Op=H6, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, niter=niter_l2, warm=True)
    l2_7_me_a = prox.L2_ncvx_tv(dims=(ny, nx), Op=H7, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, niter=niter_l2, warm=True)

    # L2 data term - Moreau envelope of isotropic TV
    l2_5_me = prox.L2_ncvx_tv(dims=(ny, nx), Op=H5, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, isotropic=True, niter=niter_l2, warm=True)
    l2_6_me = prox.L2_ncvx_tv(dims=(ny, nx), Op=H6, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, isotropic=True, niter=niter_l2, warm=True)
    l2_7_me = prox.L2_ncvx_tv(dims=(ny, nx), Op=H7, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_ulpda, isotropic=True, niter=niter_l2, warm=True)


    # L1 regularization (isotropic TV) for ULPDA or PDHG
    l1iso = pyproximal.L21(ndim=2, sigma=tau)

    # L1 regularization (anisotropic TV) for ULPDA or PDHG
    l1 = pyproximal.L1(sigma=tau)

    # Isotropic TV for MYULA or Proximal Gradient
    tv = pyproximal.TV(dims=img.shape, sigma=tau, niter=niter_tv)

    # Identity operator
    Iop = pylops.Identity(ny * nx)

    # Primal-dual
    def callback(x, f, g, K, cost, xtrue, err):
        cost.append(f(x) + g(K.matvec(x)))
        err.append(np.linalg.norm(x - xtrue))

    L = 8. / sampling ** 2 # maxeig(Gop^H Gop)
    tau0 = 0.95 / np.sqrt(L)
    mu0 = 0.95 / (tau0 * L)

    x0 = np.zeros(img.ravel().shape)
    
    ## Compute MAP estimators using Adaptive PDHG or Accelerated Proximal Gradient (FISTA)
    if computeMAP:
        cost_5_map = []
        err_5_map = []
        iml12_5_map, _ = \
            pyproximal.optimization.primaldual.AdaptivePrimalDual(l2_5, l1iso, Gop, tau=tau0, mu=mu0,
                                                        x0=x0, niter=niter_map, show=True,
                                                        callback=lambda x: callback(x, l2_5, l1iso,
                                                                                    Gop, cost_5_map,
                                                                                    img.ravel(),
                                                                                    err_5_map))
        iml12_5_map = iml12_5_map.reshape(img.shape)


        cost_6_map = []
        err_6_map = []
        iml12_6_map, _ = \
            pyproximal.optimization.primaldual.AdaptivePrimalDual(l2_6, l1iso, Gop, tau=tau0, mu=mu0,
                                                        x0=x0, niter=niter_map, show=True,
                                                        callback=lambda x: callback(x, l2_6, l1iso,
                                                                                    Gop, cost_6_map,
                                                                                    img.ravel(),
                                                                                    err_6_map))
        iml12_6_map = iml12_6_map.reshape(img.shape)


        cost_7_map = []
        err_7_map = []
        iml12_7_map, _ = \
            pyproximal.optimization.primaldual.AdaptivePrimalDual(l2_7, l1iso, Gop, tau=tau0, mu=mu0,
                                                        x0=x0, niter=niter_map, show=True,
                                                        callback=lambda x: callback(x, l2_7, l1iso,
                                                                                    Gop, cost_7_map,
                                                                                    img.ravel(),
                                                                                    err_7_map))
        iml12_7_map = iml12_7_map.reshape(img.shape)


        cost_5_mc_map = []
        err_5_mc_map = []
        iml12_5_mc_map, _ = \
            pyproximal.optimization.primaldual.AdaptivePrimalDual(l2_5_mc, l1, Gop, tau=tau0, mu=mu0,
                                                        x0=x0, niter=niter_map, show=True,
                                                        callback=lambda x: callback(x, l2_5_mc, l1,
                                                                                    Gop, cost_5_mc_map,
                                                                                    img.ravel(),
                                                                                    err_5_mc_map))
        iml12_5_mc_map = iml12_5_mc_map.reshape(img.shape)


        cost_6_mc_map = []
        err_6_mc_map = []
        iml12_6_mc_map, _ = \
            pyproximal.optimization.primaldual.AdaptivePrimalDual(l2_6_mc, l1, Gop, tau=tau0, mu=mu0,
                                                        x0=x0, niter=niter_map, show=True,
                                                        callback=lambda x: callback(x, l2_6_mc, l1,
                                                                                    Gop, cost_6_mc_map,
                                                                                    img.ravel(),
                                                                                    err_6_mc_map))
        iml12_6_mc_map = iml12_6_mc_map.reshape(img.shape)


        cost_7_mc_map = []
        err_7_mc_map = []
        iml12_7_mc_map, _ = \
            pyproximal.optimization.primaldual.AdaptivePrimalDual(l2_7_mc, l1, Gop, tau=tau0, mu=mu0,
                                                        x0=x0, niter=niter_map, show=True,
                                                        callback=lambda x: callback(x, l2_7_mc, l1,
                                                                                    Gop, cost_7_mc_map,
                                                                                    img.ravel(),
                                                                                    err_7_mc_map))
        iml12_7_mc_map = iml12_7_mc_map.reshape(img.shape)
        

        cost_5_me_map = []
        err_5_me_map = []
        iml12_5_me_map = \
            pyproximal.optimization.primal.ProximalGradient(l2_5_me, tv, x0=x0,
                                                            niter=niter_map, show=True, acceleration='fista',
                                                            callback=lambda x: callback(x, l2_5_me, tv,
                                                                                    Iop, cost_5_me_map,
                                                                                    img.ravel(),
                                                                                    err_5_me_map))
        iml12_5_me_map = iml12_5_me_map.reshape(img.shape)


        cost_6_me_map = []
        err_6_me_map = []
        iml12_6_me_map = \
            pyproximal.optimization.primal.ProximalGradient(l2_6_me, tv, x0 = x0,
                                                            niter=niter_map, show=True, acceleration='fista',
                                                            callback=lambda x: callback(x, l2_6_me, tv,
                                                                                    Iop, cost_6_me_map,
                                                                                    img.ravel(),
                                                                                    err_6_me_map))
        iml12_6_me_map = iml12_6_me_map.reshape(img.shape)


        cost_7_me_map = []
        err_7_me_map = []
        iml12_7_me_map = \
            pyproximal.optimization.primal.ProximalGradient(l2_7_me, tv, x0=x0,
                                                            niter=niter_map, show=True, acceleration='fista',
                                                            callback=lambda x: callback(x, l2_7_me, tv,
                                                                                    Iop, cost_7_me_map,
                                                                                    img.ravel(),
                                                                                    err_7_me_map))
        iml12_7_me_map = iml12_7_me_map.reshape(img.shape)

        print(f"SNR of PDHG MAP image with TV (M1): {signal_noise_ratio(img, iml12_5_map)}")
        print(f"SNR of PDHG MAP image with MC-TV (M2): {signal_noise_ratio(img, iml12_5_mc_map)}")
        print(f"SNR of PDHG MAP image with ME-TV (M3): {signal_noise_ratio(img, iml12_5_me_map)}")
        print(f"SNR of PDHG MAP image with TV (M4): {signal_noise_ratio(img, iml12_6_map)}")
        print(f"SNR of PDHG MAP image with MC-TV (M5): {signal_noise_ratio(img, iml12_6_mc_map)}")
        print(f"SNR of PDHG MAP image with ME-TV (M6): {signal_noise_ratio(img, iml12_6_me_map)}")
        print(f"SNR of PDHG MAP image with TV (M7): {signal_noise_ratio(img, iml12_7_map)}")
        print(f"SNR of PDHG MAP image with MC-TV (M8): {signal_noise_ratio(img, iml12_7_mc_map)}")
        print(f"SNR of PDHG MAP image with ME-TV (M9): {signal_noise_ratio(img, iml12_7_me_map)}")

        print(f"PSNR of PDHG MAP image with TV (M1): {psnr(img, iml12_5_map)}")
        print(f"PSNR of PDHG MAP image with MC-TV (M2): {psnr(img, iml12_5_mc_map)}")
        print(f"PSNR of PDHG MAP image with ME-TV (M3): {psnr(img, iml12_5_me_map)}")
        print(f"PSNR of PDHG MAP image with TV (M4): {psnr(img, iml12_6_map)}")
        print(f"PSNR of PDHG MAP image with MC-TV (M5): {psnr(img, iml12_6_mc_map)}")
        print(f"PSNR of PDHG MAP image with ME-TV (M6): {psnr(img, iml12_6_me_map)}")
        print(f"PSNR of PDHG MAP image with TV (M7): {psnr(img, iml12_7_map)}")
        print(f"PSNR of PDHG MAP image with MC-TV (M8): {psnr(img, iml12_7_mc_map)}")
        print(f"PSNR of PDHG MAP image with ME-TV (M9): {psnr(img, iml12_7_me_map)}")

        fig2, axes = plt.subplots(2, 5, figsize=(20, 8))
        plt.gray()  # show the filtered result in grayscale
        axes[0,0].imshow(img)
        axes[0,0].set_title("Ground truth", fontsize=16)

        # axes[0,0].imshow(y)
        # axes[0,0].set_title("Blurred and noisy image", fontsize=16)

        axes[0,1].imshow(iml12_5_map)        
        axes[0,1].set_title(r"$\mathcal{M}_1$ ($\mathbf{H}_1$, TV)", fontsize=16)

        axes[0,2].imshow(iml12_5_mc_map)
        axes[0,2].set_title(r"$\mathcal{M}_2$ ($\mathbf{H}_1$, MC-TV)", fontsize=16)

        axes[0,3].imshow(iml12_5_me_map)
        axes[0,3].set_title(r"$\mathcal{M}_3$ ($\mathbf{H}_1$, ME-TV)", fontsize=16)

        axes[0,4].imshow(iml12_6_map)
        axes[0,4].set_title(r"$\mathcal{M}_4$ ($\mathbf{H}_2$, TV)", fontsize=16)

        axes[1,0].imshow(iml12_6_mc_map)
        axes[1,0].set_title(r"$\mathcal{M}_5$ ($\mathbf{H}_2$, MC-TV)", fontsize=16)

        axes[1,1].imshow(iml12_6_me_map)
        axes[1,1].set_title(r"$\mathcal{M}_6$ ($\mathbf{H}_2$, ME-TV)", fontsize=16)

        axes[1,2].imshow(iml12_7_map)
        axes[1,2].set_title(r"$\mathcal{M}_7$ ($\mathbf{H}_3$, TV)", fontsize=16)

        axes[1,3].imshow(iml12_7_mc_map)
        axes[1,3].set_title(r"$\mathcal{M}_8$ ($\mathbf{H}_3$, MC-TV)", fontsize=16)

        axes[1,4].imshow(iml12_7_me_map)
        axes[1,4].set_title(r"$\mathcal{M}_9$ ($\mathbf{H}_3$, ME-TV)", fontsize=16)

        # plt.show()
        plt.show(block=False)
        plt.pause(10)
        plt.close()
        fig2.savefig(f'./fig/fig_prox_lmc_deconv_{image}_{niter_map}_map.pdf', dpi=500) 
    


    # Generate samples using ULPDA or MYULA
    cost_5_samples = []
    err_5_samples = []
    iml12_5_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_5, l1iso, Gop, 
                                            tau=tau0, mu=mu0, theta=1., 
                                            x0=x0, gfirst=False, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_5, l1iso, 
                                                                        Gop, cost_5_samples,
                                                                        img.ravel(), 
                                                                        err_5_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_5, tv, tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_5, tv,
                                                                        Iop, cost_5_samples,
                                                                        img.ravel(),
                                                                        err_5_samples))
        

    cost_6_samples = []
    err_6_samples = []
    iml12_6_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_6, l1iso, Gop,
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, gfirst=False, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_6, l1iso,
                                                                        Gop, cost_6_samples,
                                                                        img.ravel(),
                                                                        err_6_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_6, tv, tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_6, tv,
                                                                        Iop, cost_6_samples,
                                                                        img.ravel(),
                                                                        err_6_samples))
    

    cost_7_samples = []
    err_7_samples = []
    iml12_7_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_7, l1iso, Gop,
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, gfirst=False, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_7, l1iso,
                                                                        Gop, cost_7_samples,
                                                                        img.ravel(),
                                                                        err_7_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_7, tv, tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_7, tv,
                                                                        Iop, cost_7_samples,
                                                                        img.ravel(),
                                                                        err_7_samples))

    
    cost_5_mc_samples = []
    err_5_mc_samples = []
    iml12_5_mc_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_5_mc, l1, Gop,
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, gfirst=False, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_5_mc, l1,
                                                                        Gop, cost_5_mc_samples,
                                                                        img.ravel(),
                                                                        err_5_mc_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_5_mc, tv, tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_5_mc, tv,
                                                                        Iop, cost_5_mc_samples,
                                                                        img.ravel(),
                                                                        err_5_mc_samples))
    
    cost_6_mc_samples = []
    err_6_mc_samples = []
    iml12_6_mc_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_6_mc, l1, Gop,
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, gfirst=False, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_6_mc, l1,
                                                                        Gop, cost_6_mc_samples,
                                                                        img.ravel(),
                                                                        err_6_mc_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_6_mc, tv, tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_6_mc, tv,
                                                                        Iop, cost_6_mc_samples,
                                                                        img.ravel(),
                                                                        err_6_mc_samples))
        
    cost_7_mc_samples = []
    err_7_mc_samples = []
    iml12_7_mc_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_7_mc, l1, Gop,
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, gfirst=False, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_7_mc, l1,
                                                                        Gop, cost_7_mc_samples,
                                                                        img.ravel(),
                                                                        err_7_mc_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_7_mc, tv, tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_7_mc, tv,
                                                                        Iop, cost_7_mc_samples,
                                                                        img.ravel(),
                                                                        err_7_mc_samples))
    


    cost_5_me_samples = []
    err_5_me_samples = []
    iml12_5_me_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_5_me, l1iso, Gop,
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, gfirst=False, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_5_me, l1iso,
                                                                        Gop, cost_5_me_samples,
                                                                        img.ravel(),
                                                                        err_5_me_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_5_me, tv, tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_5_me, tv,
                                                                        Iop, cost_5_me_samples,
                                                                        img.ravel(),
                                                                        err_5_me_samples))
    
    cost_6_me_samples = []
    err_6_me_samples = []
    iml12_6_me_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_6_me, l1iso, Gop,
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, gfirst=False, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_6_me, l1iso,
                                                                        Gop, cost_6_me_samples,
                                                                        img.ravel(),
                                                                        err_6_me_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_6_me, tv, tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True, seed=seed,
                                            callback=lambda x: callback(x, l2_6_me, tv,
                                                                        Iop, cost_6_me_samples,
                                                                        img.ravel(),
                                                                        err_6_me_samples))
        
    cost_7_me_samples = []
    err_7_me_samples = []
    iml12_7_me_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_7_me, l1iso, Gop,
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, gfirst=False, niter=N, show=True,
                                            callback=lambda x: callback(x, l2_7_me, l1iso,
                                                                        Gop, cost_7_me_samples,
                                                                        img.ravel(),
                                                                        err_7_me_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_7_me, tv, 
                                            tau=tau0, mu=mu0, theta=1.,
                                            x0=x0, niter=N, show=True,
                                            callback=lambda x: callback(x, l2_7_me, tv,
                                                                        Iop, cost_7_me_samples,
                                                                        img.ravel(),
                                                                        err_7_me_samples))
    

    # Compute SNR, PSNR and MSE of samples (Require the ground truth image which might not be available in practice)
    print(f"SNR of {alg} posterior mean image with TV (M1): {signal_noise_ratio(img.ravel(), iml12_5_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with MC-TV (M2): {signal_noise_ratio(img.ravel(), iml12_5_mc_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with ME-TV (M3): {signal_noise_ratio(img.ravel(), iml12_5_me_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with TV (M4): {signal_noise_ratio(img.ravel(), iml12_6_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with MC-TV (M5): {signal_noise_ratio(img.ravel(), iml12_6_mc_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with ME-TV (M6): {signal_noise_ratio(img.ravel(), iml12_6_me_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with TV (M7): {signal_noise_ratio(img.ravel(), iml12_7_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with MC-TV (M8): {signal_noise_ratio(img.ravel(), iml12_7_mc_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with ME-TV (M9): {signal_noise_ratio(img.ravel(), iml12_7_me_samples.mean(axis=0))}")


    print(f"PSNR of {alg} posterior mean image with TV (M1): {psnr(img.ravel(), iml12_5_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with MC-TV (M2): {psnr(img.ravel(), iml12_5_mc_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with ME-TV (M3): {psnr(img.ravel(), iml12_5_me_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with TV (M4): {psnr(img.ravel(), iml12_6_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with MC-TV (M5): {psnr(img.ravel(), iml12_6_mc_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with ME-TV (M6): {psnr(img.ravel(), iml12_6_me_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with TV (M7): {psnr(img.ravel(), iml12_7_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with MC-TV (M8): {psnr(img.ravel(), iml12_7_mc_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with ME-TV (M9): {psnr(img.ravel(), iml12_7_me_samples.mean(axis=0))}")


    print(f"MSE of {alg} posterior mean image with TV (M1): {mse(img.ravel(), iml12_5_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with MC-TV (M2): {mse(img.ravel(), iml12_5_mc_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with ME-TV (M3): {mse(img.ravel(), iml12_5_me_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with TV (M4): {mse(img.ravel(), iml12_6_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with MC-TV (M5): {mse(img.ravel(), iml12_6_mc_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with ME-TV (M6): {mse(img.ravel(), iml12_6_me_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with TV (M7): {mse(img.ravel(), iml12_7_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with MC-TV (M8): {mse(img.ravel(), iml12_7_mc_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with ME-TV (M9): {mse(img.ravel(), iml12_7_me_samples.mean(axis=0))}")


    # Plot the results
    fig3, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.gray()  # show the filtered result in grayscale
    # axes[0,0].imshow(img)
    # axes[0,0].set_title("True image", fontsize=16)

    axes[0,0].imshow(y)
    axes[0,0].set_title("Blurred and noisy image", fontsize=16)

    axes[0,1].imshow(iml12_5_samples.mean(axis=0).reshape(img.shape))
    axes[0,1].set_title(r"$\mathcal{M}_1$ ($\mathbf{H}_1$, TV)", fontsize=16)

    axes[0,2].imshow(iml12_5_mc_samples.mean(axis=0).reshape(img.shape))
    axes[0,2].set_title(r"$\mathcal{M}_2$ ($\mathbf{H}_1$, MC-TV)", fontsize=16)

    axes[0,3].imshow(iml12_5_me_samples.mean(axis=0).reshape(img.shape))
    axes[0,3].set_title(r"$\mathcal{M}_3$ ($\mathbf{H}_1$, ME-TV)", fontsize=16)

    axes[0,4].imshow(iml12_6_samples.mean(axis=0).reshape(img.shape))
    axes[0,4].set_title(r"$\mathcal{M}_4$ ($\mathbf{H}_2$, TV)", fontsize=16)

    axes[1,0].imshow(iml12_6_mc_samples.mean(axis=0).reshape(img.shape))
    axes[1,0].set_title(r"$\mathcal{M}_5$ ($\mathbf{H}_2$, MC-TV)", fontsize=16)

    axes[1,1].imshow(iml12_6_me_samples.mean(axis=0).reshape(img.shape))
    axes[1,1].set_title(r"$\mathcal{M}_6$ ($\mathbf{H}_2$, ME-TV)", fontsize=16)

    axes[1,2].imshow(iml12_7_samples.mean(axis=0).reshape(img.shape))
    axes[1,2].set_title(r"$\mathcal{M}_7$ ($\mathbf{H}_3$, TV)", fontsize=16)

    axes[1,3].imshow(iml12_7_mc_samples.mean(axis=0).reshape(img.shape))
    axes[1,3].set_title(r"$\mathcal{M}_8$ ($\mathbf{H}_3$, MC-TV)", fontsize=16)

    axes[1,4].imshow(iml12_7_me_samples.mean(axis=0).reshape(img.shape))
    axes[1,4].set_title(r"$\mathcal{M}_9$ ($\mathbf{H}_3$, ME-TV)", fontsize=16)


    # plt.show()
    plt.show(block=False)
    plt.pause(10)
    plt.close()
    fig3.savefig(f'./fig/fig_prox_lmc_deconv_{image}_{alg}_{N}_3.pdf', dpi=500)

    def U(x, f, g, Op=None):
        return f(x) + g(Op.matvec(x)) if Op is not None else f(x) + g(x)


    def truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                                        iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                                        iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples, 
                                        alpha=0.8):
        # burnin_len = int(burnin * len(iml12_5_samples))

        U5 = lambda sample: U(sample, l2_5, l1iso, Gop)
        U6 = lambda sample: U(sample, l2_6, l1iso, Gop)
        U7 = lambda sample: U(sample, l2_7, l1iso, Gop)
        U5_mc = lambda sample: U(sample, l2_5_mc, l1, Gop)
        U6_mc = lambda sample: U(sample, l2_6_mc, l1, Gop)
        U7_mc = lambda sample: U(sample, l2_7_mc, l1, Gop)
        U5_me = lambda sample: U(sample, l2_5_me, l1iso, Gop)
        U6_me = lambda sample: U(sample, l2_6_me, l1iso, Gop)
        U7_me = lambda sample: U(sample, l2_7_me, l1iso, Gop)
        U_list = [U5, U6, U7, U5_mc, U6_mc, U7_mc, U5_me, U6_me, U7_me]
        
        neg_log_posteriors_5 = np.array([U5(sample) for sample in iml12_5_samples])
        neg_log_posteriors_6 = np.array([U6(sample) for sample in iml12_6_samples])
        neg_log_posteriors_7 = np.array([U7(sample) for sample in iml12_7_samples])
        neg_log_posteriors_5_mc = np.array([U5_mc(sample) for sample in iml12_5_mc_samples])
        neg_log_posteriors_6_mc = np.array([U6_mc(sample) for sample in iml12_6_mc_samples])
        neg_log_posteriors_7_mc = np.array([U7_mc(sample) for sample in iml12_7_mc_samples])
        neg_log_posteriors_5_me = np.array([U5_me(sample) for sample in iml12_5_me_samples])
        neg_log_posteriors_6_me = np.array([U6_me(sample) for sample in iml12_6_me_samples])
        neg_log_posteriors_7_me = np.array([U7_me(sample) for sample in iml12_7_me_samples])
        neg_log_posteriors = np.vstack((neg_log_posteriors_5, neg_log_posteriors_6, neg_log_posteriors_7,
                                        neg_log_posteriors_5_mc, neg_log_posteriors_6_mc, neg_log_posteriors_7_mc,
                                        neg_log_posteriors_5_me, neg_log_posteriors_6_me, neg_log_posteriors_7_me))
        
        etas = np.quantile(neg_log_posteriors, 1 - alpha, axis=1) # compute the HPD thresholds
        neg_log_posteriors_5s = np.array([[U(sample) for U in U_list] for sample in iml12_5_samples])
        neg_log_posteriors_6s = np.array([[U(sample) for U in U_list] for sample in iml12_6_samples])
        neg_log_posteriors_7s = np.array([[U(sample) for U in U_list] for sample in iml12_7_samples])
        neg_log_posteriors_5s_mc = np.array([[U(sample) for U in U_list] for sample in iml12_5_mc_samples])
        neg_log_posteriors_6s_mc = np.array([[U(sample) for U in U_list] for sample in iml12_6_mc_samples])
        neg_log_posteriors_7s_mc = np.array([[U(sample) for U in U_list] for sample in iml12_7_mc_samples])
        neg_log_posteriors_5s_me = np.array([[U(sample) for U in U_list] for sample in iml12_5_me_samples])
        neg_log_posteriors_6s_me = np.array([[U(sample) for U in U_list] for sample in iml12_6_me_samples])
        neg_log_posteriors_7s_me = np.array([[U(sample) for U in U_list] for sample in iml12_7_me_samples])
        ind_5s = (neg_log_posteriors_5s <= etas).any(axis=1)
        ind_6s = (neg_log_posteriors_6s <= etas).any(axis=1)
        ind_7s = (neg_log_posteriors_7s <= etas).any(axis=1)
        ind_5s_mc = (neg_log_posteriors_5s_mc <= etas).any(axis=1)
        ind_6s_mc = (neg_log_posteriors_6s_mc <= etas).any(axis=1)
        ind_7s_mc = (neg_log_posteriors_7s_mc <= etas).any(axis=1)
        ind_5s_me = (neg_log_posteriors_5s_me <= etas).any(axis=1)
        ind_6s_me = (neg_log_posteriors_6s_me <= etas).any(axis=1)
        ind_7s_me = (neg_log_posteriors_7s_me <= etas).any(axis=1)
        inds = np.vstack((ind_5s, ind_6s, ind_7s, ind_5s_mc, ind_6s_mc, ind_7s_mc, ind_5s_me, ind_6s_me, ind_7s_me))

        log_weights = -neg_log_posteriors - np.max(-neg_log_posteriors, axis=1)[:, np.newaxis]
        weights = np.exp(log_weights)
        truncated_weights = np.where(inds, 1 / weights, 0)        
        marginal_likelihoods = 1 / np.mean(truncated_weights, axis=1)
        marginal_posteriors = marginal_likelihoods / np.sum(marginal_likelihoods)
        return marginal_likelihoods, marginal_posteriors

    marginal_likelihoods, marginal_posteriors = truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                    iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                    iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples,
                    alpha)
    print(marginal_likelihoods)
    print(marginal_posteriors)


    def bayes_factor(marginal_likelihoods):
        res = []
        for i in range(marginal_likelihoods.shape[0]):
            for j in range(i + 1, marginal_likelihoods.shape[0]):
                res.append(marginal_likelihoods[i] / marginal_likelihoods[j])
        return res
       
    
    bfs = bayes_factor(marginal_likelihoods)
    print(bfs)



if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(prox_lmc_deconv)