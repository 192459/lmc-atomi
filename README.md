# Non-Log-Concave and Nonsmooth Sampling via Langevin Monte Carlo Algorithms

## Install Required Python Libraries
```bash
pip install -U numpy matplotlib scipy seaborn fire fastprogress SciencePlots scikit-image pylops pyproximal
```


## Mixtures of Gaussians
```bash
python lmc.py --gamma_ula=7.5e-2 --gamma_mala=7.5e-2 --gamma_pula=7.5e-2 --gamma_ihpula=2.5e-2 --gamma_mla=7.5e-2 --K=10000 --n=5
```


## Mixtures of Laplacians
```bash
python lmc_laplace.py --gamma_ula=5e-2 --gamma_mala=5e-2 --gamma_pula=5e-2 --gamma_mla=5e-2 --lamda=1e-1 --alpha=5e-1 --n=5 --K=50000
python lmc_laplace.py --gamma_ula=5e-2 --gamma_mala=5e-2 --gamma_pula=5e-2 --gamma_mla=5e-2 --lamda=5e-1 --alpha=5e-1 --n=5 --K=50000
python lmc_laplace.py --gamma_ula=5e-2 --gamma_mala=5e-2 --gamma_pula=5e-2 --gamma_mla=5e-2 --lamda=1e0 --alpha=5e-1 --n=5 --K=50000

python lmc_laplace.py --gamma_ula=1e-1 --gamma_mala=1e-1 --gamma_pula=1e-1 --gamma_mla=1e-1 --lamda=1e0 --alpha=5e-1 --n=1 --K=50000
python lmc_laplace.py --gamma_ula=1e-1 --gamma_mala=1e-1 --gamma_pula=1e-1 --gamma_mla=1e-1 --lamda=1e0 --alpha=5e-1 --n=2 --K=50000
python lmc_laplace.py --gamma_ula=1e-1 --gamma_mala=1e-1 --gamma_pula=1e-1 --gamma_mla=1e-1 --lamda=1e0 --alpha=5e-1 --n=3 --K=50000
python lmc_laplace.py --gamma_ula=1e-1 --gamma_mala=1e-1 --gamma_pula=1e-1 --gamma_mla=1e-1 --lamda=1e0 --alpha=5e-1 --n=4 --K=50000
python lmc_laplace.py --gamma_ula=1e-1 --gamma_mala=1e-1 --gamma_pula=1e-1 --gamma_mla=1e-1 --lamda=1e0 --alpha=5e-1 --n=5 --K=50000


```


## Mixtures of Gaussians with Laplacian Priors
```bash
python prox_lmc.py --gamma_pgld=5e-3 --gamma_myula=5e-3 --gamma_mymala=5e-3 --gamma_ppula=5e-3 --gamma_fbula=5e-3 --gamma_lbmumla=5e-3 --gamma0_ulpda=5e-3 --gamma1_ulpda=5e-3 --alpha=1.5e-1 --lamda=2.5e-1 --K=50000 --n=1

python prox_lmc.py --gamma_pgld=8e-2 --gamma_myula=8e-2 --gamma_mymala=8e-2 --gamma_ppula=8e-2 --gamma_fbula=8e-2 --gamma_lbmumla=8e-2 --gamma0_ulpda=8e-2 --gamma1_ulpda=8e-2 --alpha=1.5e-1 --lamda=2.5e-1 --t=100 --seed=0 --K=50000 --n=2


```


## Bayesian Imaging Deconvolution
### Camera Test Image
- MAP Estimator
```bash
python python prox_lmc_deconv.py --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --niter_MAP=1000 --image='camera' --compute_MAP=True
```
- Posterior Mean by ULPDA
```bash
python python prox_lmc_deconv.py --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --N=1000 --image='camera' --alg='ULPDA'
```
- Posterior Mean by MYULA
```bash
python python prox_lmc_deconv.py --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --N=1000 --image='camera' --alg='MYULA'
```

### Einstein Test Image
- MAP Estimator
```bash
python python prox_lmc_deconv.py --image='einstein' --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --niter_MAP=1000 --compute_MAP=True
```
- Posterior Mean by ULPDA
```bash
python python prox_lmc_deconv.py --image='einstein' --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --N=1000 --alg='ULPDA'
```
- Posterior Mean by MYULA
```bash
python python prox_lmc_deconv.py --image='einstein' --gamma_mc=15. --gamma_me=15. --sigma=0.75 --tau=0.3 --N=1000 --alg='MYULA'
```
