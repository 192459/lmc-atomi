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

# Usage: python sgld.py --gamma_ula=7.5e-2 --gamma_mala=7.5e-2 
# --gamma_pula=8e-2 --gamma_ihpula=5e-4 --gamma_mla=5e-2 --K=5000 --n=5

import os
import itertools
from fastprogress import progress_bar
from typing import NamedTuple
import fire

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import scienceplots
plt.style.use(['science', 'grid'])

import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as stats

import blackjax
import blackjax.sgmcmc.gradients as gradients
import optax
from blackjax.types import PyTree
from optax._src.base import OptState

import ProxNest
import ProxNest.utils as utils
import ProxNest.sampling as sampling
import ProxNest.optimisations as optimisations
import ProxNest.operators as operators


class GaussianMixtureSampling:
    def __init__(self, lamda, positions, sigma) -> None:
        self.lamda = lamda 
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(stats.multivariate_normal.logpdf(x, self.mu, self.sigma))

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
        cfset = ax.contourf(xx, yy, f, cmap=cm.viridis)
        ax.imshow(np.rot90(f), cmap=cm.viridis, extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')

        plt.rcParams['axes.titlepad'] = 15.
        plt.title("Samples from a mixture of 25 normal distributions")

        plt.show()
        # plt.pause(5)
        # plt.close()

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')

        ax1.plot_surface(xx, yy, f, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
        ax1.view_init(45, -70)
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # ax1.set_zticks([])
        # ax1.set_xlabel(r'$x_1$')
        # ax1.set_ylabel(r'$x_2$')


        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.contourf(xx, yy, f, zdir='z', offset=0, cmap=cm.viridis)
        ax2.view_init(90, 270)

        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        # ax2.set_xlabel(r'$x_1$')
        # ax2.set_ylabel(r'$x_2$')

        # plt.suptitle("True 2D Gaussian Mixture") 
        plt.show()
        return f


class SGLD:
    def __init__(self, lamda, positions, sigma) -> None:
        self.lamda = lamda 
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))

    def sampling(self, seed, num_training_steps):
        schedule_fn = lambda k: 0.05 * k ** (-0.55)
        schedule = [schedule_fn(i) for i in range(1, num_training_steps+1)]

        grad_fn = lambda x, _: jax.grad(self.logprob_fn)(x)
        sgld = blackjax.sgld(grad_fn)        

        rng_key = jax.random.PRNGKey(seed)
        init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))

        position = init_position
        sgld_samples = []
        for i in progress_bar(range(num_training_steps)):
            _, rng_key = jax.random.split(rng_key)
            position = jax.jit(sgld)(rng_key, position, 0, schedule[i])
            sgld_samples.append(position)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = [sample[0] for sample in sgld_samples]
        y = [sample[1] for sample in sgld_samples]

        ax.plot(x, y, 'k-', lw=0.1, alpha=0.5)
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])

        plt.axis('off')
        plt.show()
        # plt.show(block=False)
        # plt.pause(5)
        # plt.close()
        return sgld_samples


class cyclicalSGLD:
    def __init__(self, lamda, positions, sigma) -> None:
        self.lamda = lamda 
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def logprob_fn(self, x, *_):
        return self.lamda * jsp.special.logsumexp(jax.scipy.stats.multivariate_normal.logpdf(x, self.mu, self.sigma))

    def sampling(self, seed, num_training_steps):
        pass


class SPGLD:
    def __init__(self, lamda, positions, sigma) -> None:
        self.lamda = lamda 
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)


    def sampling(self):
        
        return 


class Prox:
    def __init__(self, lamda, positions, sigma) -> None:
        self.lamda = lamda 
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)


    def sampling(self):
        
        return 


class MYULA:
    def __init__(self, lamda, positions, sigma) -> None:
        self.lamda = lamda 
        self.positions = positions
        self.mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
        self.sigma = sigma * jnp.eye(2)

    def sampling(self):

        pass 


if __name__ == '__main__':
    lamda = 1/25
    positions = [-4, -2, 0, 2, 4]
    sigma = 0.03
    xmin, ymin = -5, -5
    xmax, ymax = 5, 5
    N = 300
    nbins = 300j
    X = np.linspace(-5, 5, N)
    Y = np.linspace(-5, 5, N)
    X, Y = np.meshgrid(X, Y)
    
    Z = GaussianMixtureSampling(lamda, positions, sigma).sampling(0, xmin, ymin, xmax, ymax, nbins)

    seed = 0 
    num_training_steps = 50000
    Z1 = SGLD(lamda, positions, sigma).sampling(seed, num_training_steps)

    # cyclicalSGLD().sampling()

    # Langevin().sampling()

    # HMC().sampling()

    # fire.Fire(Langevin())

    

    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    # fig2.suptitle("True density and KDEs of samples") 

    axes[0,0].contourf(X, Y, Z, cmap=cm.viridis)
    axes[0,0].set_title("True density")

    sns.kdeplot(x=Z1[:,0], y=Z1[:,1], cmap=cm.viridis, fill=True, thresh=0, levels=7, clip=(-5, 5), ax=axes[0,1])
    axes[0,1].set_title("SGLD")
    
    plt.show()



