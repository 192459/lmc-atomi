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

def U(x, f, g, Op=None):
        return f(x) + g(Op.matvec(x)) if Op is not None else f(x) + g(x)


def truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                                        iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                                        iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples, 
                                        alpha=0.8):
        neg_log_posteriors_5 = np.array([U(sample, l2_5, l1iso, Gop) for sample in iml12_5_samples])
        neg_log_posteriors_6 = np.array([U(sample, l2_6, l1iso, Gop) for sample in iml12_6_samples])
        neg_log_posteriors_7 = np.array([U(sample, l2_7, l1iso, Gop) for sample in iml12_7_samples])
        neg_log_posteriors_5_mc = np.array([U(sample, l2_5_mc, l1, Gop) for sample in iml12_5_mc_samples])
        neg_log_posteriors_6_mc = np.array([U(sample, l2_6_mc, l1, Gop) for sample in iml12_6_mc_samples])
        neg_log_posteriors_7_mc = np.array([U(sample, l2_7_mc, l1, Gop) for sample in iml12_7_mc_samples])
        neg_log_posteriors_5_me = np.array([U(sample, l2_5_me, l1iso, Gop) for sample in iml12_5_me_samples])
        neg_log_posteriors_6_me = np.array([U(sample, l2_6_me, l1iso, Gop) for sample in iml12_6_me_samples])
        neg_log_posteriors_7_me = np.array([U(sample, l2_7_me, l1iso, Gop) for sample in iml12_7_me_samples])
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


    def main():

        
        marginal_likelihoods, marginal_posteriors = \
            truncated_harmonic_mean_estimator(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                    iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                    iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples)
        print(marginal_likelihoods)
        print(marginal_posteriors)

        bfs = bayes_factor(iml12_5_samples, iml12_6_samples, iml12_7_samples, 
                        iml12_5_mc_samples, iml12_6_mc_samples, iml12_7_mc_samples, 
                        iml12_5_me_samples, iml12_6_me_samples, iml12_7_me_samples)
        
        print(bfs)



    

if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(main)