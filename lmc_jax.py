import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp

import fire


class Langevin:
    def __init__(self, gamma):
        self.gamma = gamma


if __name__ == '__main__':
  fire.Fire(Langevin())