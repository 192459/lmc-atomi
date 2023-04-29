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

import prox



def main(gamma_myula=5e-2, gamma_ulpda=5e-1, lamda=0.01, sigma=0.75, tau=0.03, alpha=0.8,
        N=10, niter_l2=50, niter_tv=10, niter_map=1000, image='camera', alg='ULPDA',
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

    def U(x, f, g, Op=None):
        return f(x) + g(Op.matvec(x)) if Op is not None else f(x) + g(x)


    def truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                                            # iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                                            # iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples, 
                                            alpha=0.8):
            U5 = lambda sample: U(sample, l2_5, l1iso, Gop)
            U6 = lambda sample: U(sample, l2_6, l1iso, Gop)
            U7 = lambda sample: U(sample, l2_7, l1iso, Gop)
            U5_mc = lambda sample: U(sample, l2_5_mc, l1, Gop)
            U6_mc = lambda sample: U(sample, l2_6_mc, l1, Gop)
            U7_mc = lambda sample: U(sample, l2_7_mc, l1, Gop)
            U5_me = lambda sample: U(sample, l2_5_me, l1iso, Gop)
            U6_me = lambda sample: U(sample, l2_6_me, l1iso, Gop)
            U7_me = lambda sample: U(sample, l2_7_me, l1iso, Gop)
            # U_list = [U5, U6, U7, U5_mc, U6_mc, U7_mc, U5_me, U6_me, U7_me]
            U_list = [U5, U6, U7]
            neg_log_posteriors_5 = np.array([U5(sample) for sample in iml12_5_samples])
            neg_log_posteriors_6 = np.array([U6(sample) for sample in iml12_6_samples])
            neg_log_posteriors_7 = np.array([U7(sample) for sample in iml12_7_samples])
            # neg_log_posteriors_5_mc = np.array([U5_mc(sample) for sample in iml12_5_mc_samples])
            # neg_log_posteriors_6_mc = np.array([U6_mc(sample) for sample in iml12_6_mc_samples])
            # neg_log_posteriors_7_mc = np.array([U7_mc(sample) for sample in iml12_7_mc_samples])
            # neg_log_posteriors_5_me = np.array([U5_me(sample) for sample in iml12_5_me_samples])
            # neg_log_posteriors_6_me = np.array([U6_me(sample) for sample in iml12_6_me_samples])
            # neg_log_posteriors_7_me = np.array([U7_me(sample) for sample in iml12_7_me_samples])
            # neg_log_posteriors = np.concatenate((neg_log_posteriors_5, neg_log_posteriors_6, neg_log_posteriors_7,
            #                                         neg_log_posteriors_5_mc, neg_log_posteriors_6_mc, neg_log_posteriors_7_mc,
            #                                         neg_log_posteriors_5_me, neg_log_posteriors_6_me, neg_log_posteriors_7_me), axis=1)
            neg_log_posteriors = np.vstack((neg_log_posteriors_5, neg_log_posteriors_6, neg_log_posteriors_7))
            etas = np.quantile(neg_log_posteriors, 1 - alpha, axis=1) # compute the HPD thresholds
            neg_log_posteriors_5s = np.array([[U(sample) for U in U_list] for sample in iml12_5_samples])
            neg_log_posteriors_6s = np.array([[U(sample) for U in U_list] for sample in iml12_6_samples])
            neg_log_posteriors_7s = np.array([[U(sample) for U in U_list] for sample in iml12_7_samples])
            # neg_log_posteriors_5s_mc = np.array([[U(sample) for U in U_list] for sample in iml12_5_mc_samples])
            # neg_log_posteriors_6s_mc = np.array([[U(sample) for U in U_list] for sample in iml12_6_mc_samples])
            # neg_log_posteriors_7s_mc = np.array([[U(sample) for U in U_list] for sample in iml12_7_mc_samples])
            # neg_log_posteriors_5s_me = np.array([[U(sample) for U in U_list] for sample in iml12_5_me_samples])
            # neg_log_posteriors_6s_me = np.array([[U(sample) for U in U_list] for sample in iml12_6_me_samples])
            # neg_log_posteriors_7s_me = np.array([[U(sample) for U in U_list] for sample in iml12_7_me_samples])
            ind_5s = (neg_log_posteriors_5s <= etas).any(axis=1)
            ind_6s = (neg_log_posteriors_6s <= etas).any(axis=1)
            ind_7s = (neg_log_posteriors_7s <= etas).any(axis=1)
            # ind_5s_mc = (neg_log_posteriors_5s_mc <= etas).any(axis=1)
            # ind_6s_mc = (neg_log_posteriors_6s_mc <= etas).any(axis=1)
            # ind_7s_mc = (neg_log_posteriors_7s_mc <= etas).any(axis=1)
            # ind_5s_me = (neg_log_posteriors_5s_me <= etas).any(axis=1)
            # ind_6s_me = (neg_log_posteriors_6s_me <= etas).any(axis=1)
            # ind_7s_me = (neg_log_posteriors_7s_me <= etas).any(axis=1)
            inds = np.vstack((ind_5s, ind_6s, ind_7s))
            log_weights = -neg_log_posteriors - np.max(-neg_log_posteriors, axis=1)[:, np.newaxis]
            weights = np.exp(log_weights)
            # print(weights)
            truncated_weights = np.where(inds, 1 / weights, 0)
            marginal_likelihoods = 1 / np.mean(truncated_weights, axis=1)
            marginal_posteriors = marginal_likelihoods / np.sum(marginal_likelihoods)
            return marginal_likelihoods, marginal_posteriors


    def bayes_factor(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                            # iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                            # iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples, 
                            alpha=0.8):
            marginal_likelihoods, _ = truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                                            # iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                                            # iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples, 
                                            alpha)
            res = []
            for i in range(marginal_likelihoods.shape[0]):
                for j in range(i + 1, marginal_likelihoods.shape[0]):
                    res.append(marginal_likelihoods[i] / marginal_likelihoods[j])
            return res



    marginal_likelihoods, marginal_posteriors = \
        truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, alpha)
    print('Marginal likelihoods:', marginal_likelihoods)
    print('Marginal posteriors:', marginal_posteriors)

    bfs = bayes_factor(iml12_5_samples, iml12_6_samples, iml12_7_samples, alpha)    
    print('Bayes factors:', bfs)



    

if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(main)