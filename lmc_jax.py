import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
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

import fire


class Langevin:
  def __init__(self, gamma) -> None:
    self.gamma = gamma


class HMC:
  def __init__(self, gamma) -> None:
    self.gamma = gamma



if __name__ == '__main__':
  fire.Fire(Langevin())