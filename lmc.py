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

# Install libraries: pip install -U numpy matplotlib scipy seaborn fire ProxNest jax blackjax optax

# Usage: python lmc_jax.py --gamma_ula=7.5e-2 --gamma_mala=7.5e-2 
# --gamma_pula=8e-2 --gamma_ihpula=5e-4 --gamma_mla=5e-2 --K=5000 --n=5

import os
import itertools
from fastprogress import progress_bar
from typing import NamedTuple
import fire

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.linalg import sqrtm

import ProxNest
import ProxNest.utils as utils
import ProxNest.sampling as sampling
import ProxNest.optimisations as optimisations
import ProxNest.operators as operators

import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp

import blackjax
import optax
from blackjax.types import PyTree
from optax._src.base import OptState


class GaussianMixtureSampling:
    def __init__(self, lamda, positions, sigma) -> None:
        self.lamda = lamda 
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))

    def sample_fn(self, rng_key):
        choose_key, sample_key = jax.random.split(rng_key)
        samples = jax.random.multivariate_normal(sample_key, self.mu, self.sigma)
        return jax.random.choice(choose_key, samples)

    def sampling(self, seed, xmin, ymin, xmax, ymax, nbins):
        rng_key = jax.random.PRNGKey(seed)
        samples = jax.vmap(self.sample_fn)(jax.random.split(rng_key, 10_000))

        x, y = samples[:, 0], samples[:, 1]
        xx, yy = np.mgrid[xmin:xmax:nbins, ymin:ymax:nbins]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        fig, ax = plt.subplots()
        cfset = ax.contourf(xx, yy, f, cmap='Blues')
        ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')

        plt.rcParams['axes.titlepad'] = 15.
        plt.title("Samples from a mixture of 25 normal distributions")

        plt.show(block=False)
        plt.pause(5)
        plt.close()


