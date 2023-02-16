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

import aesara
import aemcmc
import aehmc


class generateGaussianMixture:
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
  lmbda = 1/25
  positions = [-4, -2, 0, 2, 4]

  fire.Fire(Langevin())