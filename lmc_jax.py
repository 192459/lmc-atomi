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

import blackjax
import optax
from blackjax.types import PyTree
from optax._src.base import OptState

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

    plt.show(block=False)
    plt.pause(5)
    plt.close()


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
    plt.show(block=False)
    plt.pause(5)
    plt.close()


class cyclicalSGLD:
  def __init__(self, gamma) -> None:
    self.gamma = gamma


  def sampling(self, seed, num_training_steps):
    pass


class Langevin:
  def __init__(self, gamma) -> None:
    self.gamma = gamma


  def sampling(self):
    
    return 



class HMC:
  def __init__(self, gamma) -> None:
    self.gamma = gamma

  def sampling(self):

    pass 


if __name__ == '__main__':
  lamda = 1/25
  positions = [-4, -2, 0, 2, 4]
  sigma = 0.03
  xmin, ymin = -5, -5
  xmax, ymax = 5, 5
  nbins = 300j
  GaussianMixtureSampling(lamda, positions, sigma).sampling(0, xmin, ymin, xmax, ymax, nbins)

  SGLD().sampling(3, 50000)

  # cyclicalSGLD().sampling()

  # Langevin().sampling()

  # HMC().sampling()

  # fire.Fire(Langevin())