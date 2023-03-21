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

# Install libraries: pip install -U numpy matplotlib scipy seaborn fire

# To install JAX, see its documentations


'''
Usage: python lmc_laplace_jax.py --gamma_ula=5e-2 --gamma_mala=5e-2 --gamma_pula=5e-2 --gamma_ihpula=5e-2 --gamma_mla=5e-2 --nChains=4 --K=10000 --n=5 --seed=0
'''


import os
import itertools
from fastprogress import progress_bar
from typing import NamedTuple
import multiprocessing
import fire

import numpy as np
import random

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())
import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    } 
    )

if jax.lib.xla_bridge.get_backend().platform == "cpu":
    from jax.scipy.linalg import sqrtm
else:
    def sqrtm(x):
        u, s, vh = jnp.linalg.svd(x)
        return (u * s**.5) @ vh
    
import prox_jax


class LangevinMonteCarloLaplacian:
    def __init__(self, mus, alphas, omegas, lamda, K=1000, seed=0) -> None:
        super(LangevinMonteCarloLaplacian, self).__init__()
        self.mus = mus
        self.alphas = alphas
        self.omegas = omegas
        self.lamda = lamda        
        self.n = K
        self.seed = seed
        self.d = mus[0].shape[0]  

    

