python lmc.py --gamma_ula=7.5e-2 --gamma_mala=7.5e-2 --gamma_pula=7.5e-2 --gamma_ihpula=2.5e-2 --gamma_mla=7.5e-2 --K=10000 --n=5

python prox_lmc.py --gamma_proxula=5e-3 --gamma_myula=5e-3 --gamma_mymala=5e-3 --gamma_ppula=5e-3 \
--gamma_eula=5e-3 --gamma_lbmumla=5e-3 --alpha=1.5e-1 --lamda=2.5e-1 --K=50000 --n=1

CUDA_VISIBLE_DEVICES=0 python sgld.py --n=50000 --num_partitions=10000 --lr=1e-4

