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
# pip install -U numpy matplotlib scipy seaborn fire fastprogress SciencePlots scikit-image pylops pyproximal

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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import pylops
import pyproximal

import ProxNest.utils as utils
import ProxNest.sampling as sampling
import ProxNest.optimisations as optimisations
import ProxNest.operators as operators

import prox


def proxnest_deconv(gamma_pgld=5e-2, gamma_myula=5e-2, gamma_mymala=5e-2, 
                    gamma_pdhg=5e-1, gamma0_ulpda=5e-2, gamma1_ulpda=5e-2, 
                    lamda=0.01, sigma=0.47, tau=0.03, K=10000, seed=0):

    # img = data.camera()
    img = io.imread("fig/einstein.png")
    ny, nx = img.shape
    rng = default_rng(seed)

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



if __name__ == '__main__':
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fire.Fire(proxnest_deconv)
