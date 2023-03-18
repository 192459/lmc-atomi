python lmc.py --gamma_ula=7.5e-2 --gamma_mala=7.5e-2 --gamma_pula=7.5e-2 --gamma_ihpula=2.5e-2 --gamma_mla=7.5e-2 --K=10000 --n=5

python prox_lmc.py --gamma_pgld=5e-3 --gamma_myula=5e-3 --gamma_mymala=5e-3 --gamma_ppula=5e-3 \
--gamma_fbula=5e-3 --gamma_lbmumla=5e-3 --alpha=1.5e-1 --lamda=2.5e-1 --K=50000 --n=1

CUDA_VISIBLE_DEVICES=0 python sgld.py --n=50000 --num_partitions=10000 --lr=1e-4

python lmc_laplace.py --gamma_ula=1.2e-1 --gamma_mala=1.2e-1 --gamma_pula=1.2e-1 --gamma_mla=1.2e-1 --lamda=1e0 --alpha=5e-1 --n=1 --K=50000 --seed=0
python lmc_laplace.py --gamma_ula=1.2e-1 --gamma_mala=1.2e-1 --gamma_pula=1.2e-1 --gamma_mla=1.2e-1 --lamda=1e0 --alpha=5e-1 --n=2 --K=50000 --seed=0
python lmc_laplace.py --gamma_ula=1.2e-1 --gamma_mala=1.2e-1 --gamma_pula=1.2e-1 --gamma_mla=1.2e-1 --lamda=1e0 --alpha=5e-1 --n=3 --K=50000 --seed=0
python lmc_laplace.py --gamma_ula=1.2e-1 --gamma_mala=1.2e-1 --gamma_pula=1.2e-1 --gamma_mla=1.2e-1 --lamda=1e0 --alpha=5e-1 --n=4 --K=50000 --seed=0
python lmc_laplace.py --gamma_ula=8e-2 --gamma_mala=8e-2 --gamma_pula=8e-2 --gamma_mla=8e-2 --lamda=5e-1 --alpha=5e-1 --n=5 --K=80000 --seed=0

