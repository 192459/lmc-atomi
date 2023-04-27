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
def prox_lmc_deconv(gamma_myula=5e-2, gamma_pdhg=5e-1, lamda=0.01, 
                    snr=50., tau=0.03, N=10000, image='camera', alg='ULPDA', seed=0):

    # Choose the test image
    if image == 'einstein':
        img = io.imread("fig/einstein.png")
    elif image == 'camera':
        img = data.camera()
        
    ny, nx = img.shape
    rng = default_rng(seed)
    sigma = np.linalg.norm(img.ravel(), np.inf) * 10**(-snr/20)

    ###
    h5 = np.ones((5, 5))
    h5 /= h5.sum()
    nh5 = h5.shape
    H5 = pylops.signalprocessing.Convolve2D((ny, nx), h=h5, offset=(nh5[0] // 2, nh5[1] // 2))
    y = H5 * img + rng.normal(0, sigma, size=(ny, nx))
    # y5 = img + rng.normal(0, sigma, size=(ny, nx))

    h6 = np.ones((6, 6))
    h6 /= h6.sum()
    nh6 = h6.shape
    H6 = pylops.signalprocessing.Convolve2D((ny, nx), h=h6, offset=(nh6[0] // 2, nh6[1] // 2))
    # y6 = H6 * img + rng.normal(0, sigma, size=(ny, nx))

    h7 = np.ones((7, 7))
    h7 /= h7.sum()
    nh7 = h7.shape
    H7 = pylops.signalprocessing.Convolve2D((ny, nx), h=h7, offset=(nh7[0] // 2, nh7[1] // 2))
    # y7 = H7 * img + rng.normal(0, sigma, size=(ny, nx))


    # Plot of the original image and the blurred image
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    plt.gray()  # show the filtered result in grayscale
    axes[0].imshow(img)
    axes[1].imshow(y)
    plt.show(block=False)
    plt.pause(5)
    plt.close() 


    # Gradient operator
    sampling = 1.
    Gop = pylops.Gradient(dims=(ny, nx), sampling=sampling, edge=False, kind='forward', dtype='float64')
    L = 8. / sampling ** 2 # maxeig(Gop^H Gop)

    # L2 data term
    l2_5 = pyproximal.L2(Op=H5, b=y.ravel(), sigma=1/sigma**2, niter=50, warm=True)
    l2_6 = pyproximal.L2(Op=H6, b=y.ravel(), sigma=1/sigma**2, niter=50, warm=True)
    l2_7 = pyproximal.L2(Op=H7, b=y.ravel(), sigma=1/sigma**2, niter=50, warm=True)

    # L2 data term - Moreau envelope of isotropic TV
    l2_5_mc = prox.L2_minimax_concave(dims=(ny, nx), Op=H5, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_pdhg, niter=50, warm=True)
    l2_6_mc = prox.L2_minimax_concave(dims=(ny, nx), Op=H6, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_pdhg, niter=50, warm=True)
    l2_7_mc = prox.L2_minimax_concave(dims=(ny, nx), Op=H7, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_pdhg, niter=50, warm=True)

    # L2 data term - Moreau envelope of isotropic TV
    l2_5_me = prox.L2_moreau_env(dims=(ny, nx), Op=H5, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_pdhg, niter=50, warm=True)
    l2_6_me = prox.L2_moreau_env(dims=(ny, nx), Op=H6, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_pdhg, niter=50, warm=True)
    l2_7_me = prox.L2_moreau_env(dims=(ny, nx), Op=H7, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_pdhg, niter=50, warm=True)

    # L1 regularization (isotropic TV)
    l1iso = pyproximal.L21(ndim=2, sigma=tau)

    # Primal-dual
    def callback(x, f, g, K, cost, xtrue, err):
        cost.append(f(x) + g(K.matvec(x)))
        err.append(np.linalg.norm(x - xtrue))

    tau0 = 0.95 / np.sqrt(L)
    mu0 = 0.95 / np.sqrt(L)


    '''
    ## Compute MAP estimators using PDHG
    cost_5_fixed = []
    err_5_fixed = []
    iml12_5_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_5, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_5, l1iso,
                                                                                Gop, cost_5_fixed,
                                                                                img.ravel(),
                                                                                err_5_fixed))
    iml12_5_fixed = iml12_5_fixed.reshape(img.shape)


    cost_6_fixed = []
    err_6_fixed = []
    iml12_6_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_6, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6, l1iso,
                                                                                Gop, cost_6_fixed,
                                                                                img.ravel(),
                                                                                err_6_fixed))
    iml12_6_fixed = iml12_6_fixed.reshape(img.shape)


    cost_7_fixed = []
    err_7_fixed = []
    iml12_7_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_7, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7, l1iso,
                                                                                Gop, cost_7_fixed,
                                                                                img.ravel(),
                                                                                err_7_fixed))
    iml12_7_fixed = iml12_7_fixed.reshape(img.shape)


    cost_5_mc_fixed = []
    err_5_mc_fixed = []
    iml12_5_mc_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_5_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_5_mc, l1iso,
                                                                                Gop, cost_5_mc_fixed,
                                                                                img.ravel(),
                                                                                err_5_mc_fixed))
    iml12_5_mc_fixed = iml12_5_mc_fixed.reshape(img.shape)


    cost_6_mc_fixed = []
    err_6_mc_fixed = []
    iml12_6_mc_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_6_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6_mc, l1iso,
                                                                                Gop, cost_6_mc_fixed,
                                                                                img.ravel(),
                                                                                err_6_mc_fixed))
    iml12_6_mc_fixed = iml12_6_mc_fixed.reshape(img.shape)


    cost_7_mc_fixed = []
    err_7_mc_fixed = []
    iml12_7_mc_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_7_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7_mc, l1iso,
                                                                                Gop, cost_7_mc_fixed,
                                                                                img.ravel(),
                                                                                err_7_mc_fixed))
    iml12_7_mc_fixed = iml12_7_mc_fixed.reshape(img.shape)


    cost_5_me_fixed = []
    err_5_me_fixed = []
    iml12_5_me_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_5_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_5_me, l1iso,
                                                                                Gop, cost_5_me_fixed,
                                                                                img.ravel(),
                                                                                err_5_me_fixed))
    iml12_5_me_fixed = iml12_5_me_fixed.reshape(img.shape)


    cost_6_me_fixed = []
    err_6_me_fixed = []
    iml12_6_me_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_6_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6_me, l1iso,
                                                                                Gop, cost_6_me_fixed,
                                                                                img.ravel(),
                                                                                err_6_me_fixed))
    iml12_6_me_fixed = iml12_6_me_fixed.reshape(img.shape)


    cost_7_me_fixed = []
    err_7_me_fixed = []
    iml12_7_me_fixed = \
        pyproximal.optimization.primaldual.PrimalDual(l2_7_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7_me, l1iso,
                                                                                Gop, cost_7_me_fixed,
                                                                                img.ravel(),
                                                                                err_7_me_fixed))
    iml12_7_me_fixed = iml12_7_me_fixed.reshape(img.shape)

    print(f"SNR of PDHG MAP image with TV (M1): {signal_noise_ratio(img, iml12_5_fixed)}")
    print(f"SNR of PDHG MAP image with MC-TV (M2): {signal_noise_ratio(img, iml12_5_mc_fixed)}")
    print(f"SNR of PDHG MAP image with ME-TV (M3): {signal_noise_ratio(img, iml12_5_me_fixed)}")
    print(f"SNR of PDHG MAP image with TV (M4): {signal_noise_ratio(img, iml12_6_fixed)}")
    print(f"SNR of PDHG MAP image with MC-TV (M5): {signal_noise_ratio(img, iml12_6_mc_fixed)}")
    print(f"SNR of PDHG MAP image with ME-TV (M6): {signal_noise_ratio(img, iml12_6_me_fixed)}")
    print(f"SNR of PDHG MAP image with TV (M7): {signal_noise_ratio(img, iml12_7_fixed)}")
    print(f"SNR of PDHG MAP image with MC-TV (M8): {signal_noise_ratio(img, iml12_7_mc_fixed)}")
    print(f"SNR of PDHG MAP image with ME-TV (M9): {signal_noise_ratio(img, iml12_7_me_fixed)}")

    print(f"PSNR of PDHG MAP image with TV (M1): {psnr(img, iml12_5_fixed)}")
    print(f"PSNR of PDHG MAP image with MC-TV (M2): {psnr(img, iml12_5_mc_fixed)}")
    print(f"PSNR of PDHG MAP image with ME-TV (M3): {psnr(img, iml12_5_me_fixed)}")
    print(f"PSNR of PDHG MAP image with TV (M4): {psnr(img, iml12_6_fixed)}")
    print(f"PSNR of PDHG MAP image with MC-TV (M5): {psnr(img, iml12_6_mc_fixed)}")
    print(f"PSNR of PDHG MAP image with ME-TV (M6): {psnr(img, iml12_6_me_fixed)}")
    print(f"PSNR of PDHG MAP image with TV (M7): {psnr(img, iml12_7_fixed)}")
    print(f"PSNR of PDHG MAP image with MC-TV (M8): {psnr(img, iml12_7_mc_fixed)}")
    print(f"PSNR of PDHG MAP image with ME-TV (M9): {psnr(img, iml12_7_me_fixed)}")

    fig2, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.gray()  # show the filtered result in grayscale
    # axes[0,0].imshow(img)
    # axes[0,0].set_title("Ground truth", fontsize=16)

    axes[0,0].imshow(y)
    axes[0,0].set_title("Blurred and noisy image", fontsize=16)

    axes[0,1].imshow(iml12_5_fixed)
    axes[0,1].set_title(r"$\mathcal{M}_1$", fontsize=16)

    axes[0,2].imshow(iml12_5_mc_fixed)
    axes[0,2].set_title(r"$\mathcal{M}_2$", fontsize=16)

    axes[0,3].imshow(iml12_5_me_fixed)
    axes[0,3].set_title(r"$\mathcal{M}_3$", fontsize=16)

    axes[0,4].imshow(iml12_6_fixed)
    axes[0,4].set_title(r"$\mathcal{M}_4$", fontsize=16)

    axes[1,0].imshow(iml12_6_mc_fixed)
    axes[1,0].set_title(r"$\mathcal{M}_5$", fontsize=16)

    axes[1,1].imshow(iml12_6_me_fixed)
    axes[1,1].set_title(r"$\mathcal{M}_6$", fontsize=16)

    axes[1,2].imshow(iml12_7_fixed)
    axes[1,2].set_title(r"$\mathcal{M}_7$", fontsize=16)

    axes[1,3].imshow(iml12_7_mc_fixed)
    axes[1,3].set_title(r"$\mathcal{M}_8$", fontsize=16)

    axes[1,4].imshow(iml12_7_me_fixed)
    axes[1,4].set_title(r"$\mathcal{M}_9$", fontsize=16)

    # plt.show()
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    fig2.savefig(f'./fig/fig_prox_lmc_deconv_{image}_{K}_2.pdf', dpi=500) 
    '''


    # Generate samples using ULPDA or MYULA
    cost_5_samples = []
    err_5_samples = []
    iml12_5_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_5, l1iso, Gop, 
                                                    tau=tau0, mu=mu0, theta=1., 
                                                    x0=np.zeros_like(img.ravel()), 
                                                    gfirst=False, niter=N, show=True, 
                                                    callback=lambda x: callback(x, l2_5, l1iso, 
                                                                                Gop, cost_5_samples,
                                                                                img.ravel(), 
                                                                                err_5_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_5, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_5, l1iso,
                                                                                Gop, cost_5_samples,
                                                                                img.ravel(),
                                                                                err_5_samples))
        

    cost_6_samples = []
    err_6_samples = []
    iml12_6_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_6, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6, l1iso,
                                                                                Gop, cost_6_samples,
                                                                                img.ravel(),
                                                                                err_6_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_6, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6, l1iso,
                                                                                Gop, cost_6_samples,
                                                                                img.ravel(),
                                                                                err_6_samples))
    

    cost_7_samples = []
    err_7_samples = []
    iml12_7_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_7, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7, l1iso,
                                                                                Gop, cost_7_samples,
                                                                                img.ravel(),
                                                                                err_7_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_7, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7, l1iso,
                                                                                Gop, cost_7_samples,
                                                                                img.ravel(),
                                                                                err_7_samples))


    '''
    cost_5_mc_samples = []
    err_5_mc_samples = []
    iml12_5_mc_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_5_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_5_mc, l1iso,
                                                                                Gop, cost_5_mc_samples,
                                                                                img.ravel(),
                                                                                err_5_mc_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_5_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_5_mc, l1iso,
                                                                                Gop, cost_5_mc_samples,
                                                                                img.ravel(),
                                                                                err_5_mc_samples))
    
    cost_6_mc_samples = []
    err_6_mc_samples = []
    iml12_6_mc_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_6_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6_mc, l1iso,
                                                                                Gop, cost_6_mc_samples,
                                                                                img.ravel(),
                                                                                err_6_mc_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_6_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6_mc, l1iso,
                                                                                Gop, cost_6_mc_samples,
                                                                                img.ravel(),
                                                                                err_6_mc_samples))
        
    cost_7_mc_samples = []
    err_7_mc_samples = []
    iml12_7_mc_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_7_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7_mc, l1iso,
                                                                                Gop, cost_7_mc_samples,
                                                                                img.ravel(),
                                                                                err_7_mc_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_7_mc, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7_mc, l1iso,
                                                                                Gop, cost_7_mc_samples,
                                                                                img.ravel(),
                                                                                err_7_mc_samples))
    '''


    cost_5_me_samples = []
    err_5_me_samples = []
    iml12_5_me_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_5_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_5_me, l1iso,
                                                                                Gop, cost_5_me_samples,
                                                                                img.ravel(),
                                                                                err_5_me_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_5_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()), 
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_5_me, l1iso,
                                                                                Gop, cost_5_me_samples,
                                                                                img.ravel(),
                                                                                err_5_me_samples))
    
    cost_6_me_samples = []
    err_6_me_samples = []
    iml12_6_me_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_6_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6_me, l1iso,
                                                                                Gop, cost_6_me_samples,
                                                                                img.ravel(),
                                                                                err_6_me_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_6_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_6_me, l1iso,
                                                                                Gop, cost_6_me_samples,
                                                                                img.ravel(),
                                                                                err_6_me_samples))
        
    cost_7_me_samples = []
    err_7_me_samples = []
    iml12_7_me_samples = \
        prox.UnadjustedLangevinPrimalDual(l2_7_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7_me, l1iso,
                                                                                Gop, cost_7_me_samples,
                                                                                img.ravel(),
                                                                                err_7_me_samples)) if alg == 'ULPDA' else \
        prox.MoreauYosidaUnadjustedLangevin(l2_7_me, l1iso, Gop,
                                                    tau=tau0, mu=mu0, theta=1.,
                                                    x0=np.zeros_like(img.ravel()),
                                                    gfirst=False, niter=N, show=True,
                                                    callback=lambda x: callback(x, l2_7_me, l1iso,
                                                                                Gop, cost_7_me_samples,
                                                                                img.ravel(),
                                                                                err_7_me_samples))
    

    # Compute SNR, PSNR and MSE of samples (Require the ground truth image which might not be available in practice)
    print(f"SNR of {alg} posterior mean image with TV (M1): {signal_noise_ratio(img.ravel(), iml12_5_samples.mean(axis=0))}")
    # print(f"SNR of {alg} posterior mean image with MC-TV (M2): {signal_noise_ratio(img.ravel(), iml12_5_mc_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with ME-TV (M3): {signal_noise_ratio(img.ravel(), iml12_5_me_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with TV (M4): {signal_noise_ratio(img.ravel(), iml12_6_samples.mean(axis=0))}")
    # print(f"SNR of {alg} posterior mean image with MC-TV (M5): {signal_noise_ratio(img.ravel(), iml12_6_mc_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with ME-TV (M6): {signal_noise_ratio(img.ravel(), iml12_6_me_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with TV (M7): {signal_noise_ratio(img.ravel(), iml12_7_samples.mean(axis=0))}")
    # print(f"SNR of {alg} posterior mean image with MC-TV (M8): {signal_noise_ratio(img.ravel(), iml12_7_mc_samples.mean(axis=0))}")
    print(f"SNR of {alg} posterior mean image with ME-TV (M9): {signal_noise_ratio(img.ravel(), iml12_7_me_samples.mean(axis=0))}")


    print(f"PSNR of {alg} posterior mean image with TV (M1): {psnr(img.ravel(), iml12_5_samples.mean(axis=0))}")
    # print(f"PSNR of {alg} posterior mean image with MC-TV (M2): {psnr(img.ravel(), iml12_5_mc_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with ME-TV (M3): {psnr(img.ravel(), iml12_5_me_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with TV (M4): {psnr(img.ravel(), iml12_6_samples.mean(axis=0))}")
    # print(f"PSNR of {alg} posterior mean image with MC-TV (M5): {psnr(img.ravel(), iml12_6_mc_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with ME-TV (M6): {psnr(img.ravel(), iml12_6_me_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with TV (M7): {psnr(img.ravel(), iml12_7_samples.mean(axis=0))}")
    # print(f"PSNR of {alg} posterior mean image with MC-TV (M8): {psnr(img.ravel(), iml12_7_mc_samples.mean(axis=0))}")
    print(f"PSNR of {alg} posterior mean image with ME-TV (M9): {psnr(img.ravel(), iml12_7_me_samples.mean(axis=0))}")


    print(f"MSE of {alg} posterior mean image with TV (M1): {mse(img.ravel(), iml12_5_samples.mean(axis=0))}")
    # print(f"MSE of {alg} posterior mean image with MC-TV (M2): {mse(img.ravel(), iml12_5_mc_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with ME-TV (M3): {mse(img.ravel(), iml12_5_me_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with TV (M4): {mse(img.ravel(), iml12_6_samples.mean(axis=0))}")
    # print(f"MSE of {alg} posterior mean image with MC-TV (M5): {mse(img.ravel(), iml12_6_mc_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with ME-TV (M6): {mse(img.ravel(), iml12_6_me_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with TV (M7): {mse(img.ravel(), iml12_7_samples.mean(axis=0))}")
    # print(f"MSE of {alg} posterior mean image with MC-TV (M8): {mse(img.ravel(), iml12_7_mc_samples.mean(axis=0))}")
    print(f"MSE of {alg} posterior mean image with ME-TV (M9): {mse(img.ravel(), iml12_7_me_samples.mean(axis=0))}")


    # Plot the results
    fig3, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.gray()  # show the filtered result in grayscale
    # axes[0,0].imshow(img)
    # axes[0,0].set_title("True image", fontsize=16)

    axes[0,0].imshow(y)
    axes[0,0].set_title("Blurred and noisy image", fontsize=16)

    axes[0,1].imshow(iml12_5_samples.mean(axis=0).reshape(img.shape))
    axes[0,1].set_title(r"Posterior mean image ($\mathcal{M}_1$)", fontsize=16)

    # axes[0,2].imshow(iml12_5_mc_samples.mean(axis=0).reshape(img.shape))
    # axes[0,2].set_title(r"Posterior mean image ($\mathcal{M}_2$)", fontsize=16)

    axes[0,3].imshow(iml12_5_me_samples.mean(axis=0).reshape(img.shape))
    axes[0,3].set_title(r"Posterior mean image ($\mathcal{M}_3$)", fontsize=16)

    axes[0,4].imshow(iml12_6_samples.mean(axis=0).reshape(img.shape))
    axes[0,4].set_title(r"Posterior mean image ($\mathcal{M}_4$)", fontsize=16)

    # axes[1,0].imshow(iml12_6_mc_samples.mean(axis=0).reshape(img.shape))
    # axes[1,0].set_title(r"Posterior mean image ($\mathcal{M}_5$)", fontsize=16)

    axes[1,1].imshow(iml12_6_me_samples.mean(axis=0).reshape(img.shape))
    axes[1,1].set_title(r"Posterior mean image ($\mathcal{M}_6$)", fontsize=16)

    axes[1,2].imshow(iml12_7_samples.mean(axis=0).reshape(img.shape))
    axes[1,2].set_title(r"Posterior mean image ($\mathcal{M}_7$)", fontsize=16)

    # axes[1,3].imshow(iml12_7_mc_samples.mean(axis=0).reshape(img.shape))
    # axes[1,3].set_title(r"Posterior mean image ($\mathcal{M}_8$)", fontsize=16)

    axes[1,4].imshow(iml12_7_me_samples.mean(axis=0).reshape(img.shape))
    axes[1,4].set_title(r"Posterior mean image ($\mathcal{M}_9$)", fontsize=16)


    plt.show()
    # plt.show(block=False)
    # plt.pause(10)
    # plt.close()
    # fig3.savefig(f'./fig/fig_prox_lmc_deconv_{image}_{alg}_{K}_3.pdf', dpi=500)


    def U_tv(x, H): 
        neg_log_likelihood = pyproximal.L2(Op=H, b=y.ravel(), sigma=1/sigma**2, niter=50, warm=True)(x)
        neg_log_prior = l1iso(Gop.matvec(x))
        return neg_log_likelihood + neg_log_prior
    
    def U_mc(x, H): 
        neg_log_likelihood = prox.L2_minimax_concave(dims=(ny, nx), Op=H, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_pdhg, niter=50, warm=True)(x)
        neg_log_prior = l1iso(Gop.matvec(x))
        return neg_log_likelihood + neg_log_prior

    def U_me(x, H): 
        neg_log_likelihood = prox.L2_moreau_env(dims=(ny, nx), Op=H, b=y.ravel(), sigma=1/sigma**2, lamda=tau, gamma=gamma_pdhg, niter=50, warm=True)(x)
        neg_log_prior = l1iso(Gop.matvec(x))
        return neg_log_likelihood + neg_log_prior


    def truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                                        iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                                        iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples, 
                                        alpha=0.8):
        neg_log_posteriors_5 = np.array([U_tv(sample, H5) for sample in iml12_5_samples])
        neg_log_posteriors_6 = np.array([U_tv(sample, H6) for sample in iml12_6_samples])
        neg_log_posteriors_7 = np.array([U_tv(sample, H7) for sample in iml12_7_samples])
        neg_log_posteriors_5_mc = np.array([U_mc(sample, H5) for sample in iml12_5_mc_samples])
        neg_log_posteriors_6_mc = np.array([U_mc(sample, H6) for sample in iml12_6_mc_samples])
        neg_log_posteriors_7_mc = np.array([U_mc(sample, H7) for sample in iml12_7_mc_samples])
        neg_log_posteriors_5_me = np.array([U_me(sample, H5) for sample in iml12_5_me_samples])
        neg_log_posteriors_6_me = np.array([U_me(sample, H6) for sample in iml12_6_me_samples])
        neg_log_posteriors_7_me = np.array([U_me(sample, H7) for sample in iml12_7_me_samples])
        neg_log_posteriors = np.concatenate((neg_log_posteriors_5, neg_log_posteriors_6, neg_log_posteriors_7,
                                                neg_log_posteriors_5_mc, neg_log_posteriors_6_mc, neg_log_posteriors_7_mc,
                                                neg_log_posteriors_5_me, neg_log_posteriors_6_me, neg_log_posteriors_7_me), axis=1)

        eta = np.quantile(neg_log_posteriors, 1 - alpha, axis=0)     
        # samples_ind = np.zeros(samples.shape[0])
        # for k in range(samples.shape[0]):
        #     samples_ind[k] = neg_log_posteriors[k,:] <= np.max(eta)
        # samples_filtered = np.compress(samples_ind, samples, axis=0)
        # samples_filtered_U = np.zeros((samples_filtered.shape[0], len(Hs)))
        # for k in range(samples_filtered.shape[0]):
        #     for h in range(len(Hs)):
        #         samples_filtered_U[k, h] = U(samples_filtered[k], Hs[h])
        log_weights = -neg_log_posteriors
        # print(log_weights.max(axis=0))  
        weights = np.exp(log_weights - log_weights.max(axis=0))
        # print(log_weights)
        truncated_weights = np.where(neg_log_posteriors <= np.max(eta), 1 / weights, 0)
        # print(truncated_weights)
        # print(truncated_weights.shape)
        marginal_likelihoods = 1 / np.mean(truncated_weights, axis=0)
        # return np.exp(-samples_filtered_U - (-samples_U).max()).sum() / np.exp(-samples_U - (-samples_U).max()).sum()
        marginal_posteriors = marginal_likelihoods / np.sum(marginal_likelihoods)
        return marginal_likelihoods, marginal_posteriors

    # print(truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
    #                 iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
    #                 iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples))


    def bayes_factor(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                    iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                    iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples, alpha=0.8):
        marginal_likelihoods, _ = truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                                        iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                                        iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples, 
                                        alpha)
        res = []
        for i in range(marginal_likelihoods.shape[0]):
            for j in range(i + 1, marginal_likelihoods.shape[0]):
                res.append(marginal_likelihoods[i] / marginal_likelihoods[j])
        return res
    
    # print(bayes_factor(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
    #                 iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
    #                 iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples))



    '''
    prox_lmc = ProximalLangevinMonteCarloDeconvolution(lamda, sigma, tau, K, seed)      

    myula_samples = prox_lmc.myula(y, H5, gamma_myula)
    '''
    

if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(prox_lmc_deconv)