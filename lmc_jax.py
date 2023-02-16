import os
import itertools
from fastprogress import progress_bar
from typing import NamedTuple
import fire

import pandas as pd
import numpy as np
import scipy as sp
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

# import aesara
# import aemcmc
# import aehmc


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
    plt.show()
    

class Langevin:
  def __init__(self, gamma) -> None:
    self.gamma = gamma


  def unadjustedLangevin(self):
    gamma = self.gamma

    return 


class cyclicalSGLD:
    def __init__(self, gamma) -> None:
      self.gamma = gamma


class HMC:
  def __init__(self, gamma) -> None:
    self.gamma = gamma



if __name__ == '__main__':
  lamda = 1/25
  positions = [-4, -2, 0, 2, 4]
  sigma = 0.03
  xmin, ymin = -5, -5
  xmax, ymax = 5, 5
  nbins = 300j
  GaussianMixtureSampling(lamda, positions, sigma).sampling(0, xmin, ymin, xmax, ymax, nbins)

  # fire.Fire(Langevin())