CUDA_VISIBLE_DEVICES=0 nohup python train_kronecker.py --batch_size 64 --n_components 1 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.01  >> out_run01.log &
CUDA_VISIBLE_DEVICES=1 nohup python train_kronecker.py --batch_size 64 --n_components 2 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.01  >> out_run02.log &
CUDA_VISIBLE_DEVICES=2 nohup python train_kronecker.py --batch_size 64 --n_components 3 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.01  >> out_run03.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_kronecker.py --batch_size 64 --n_components 4 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.01  >> out_run04.log &

CUDA_VISIBLE_DEVICES=0 nohup python train_kronecker.py --batch_size 64 --n_components 5 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.01  >> out_run05.log &
CUDA_VISIBLE_DEVICES=1 nohup python train_kronecker.py --batch_size 64 --n_components 6 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.01  >> out_run06.log &
CUDA_VISIBLE_DEVICES=2 nohup python train_kronecker.py --batch_size 64 --n_components 7 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.01  >> out_run07.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_kronecker.py --batch_size 64 --n_components 10 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.01  >> out_run08.log &

CUDA_VISIBLE_DEVICES=0 nohup python train_kronecker.py --batch_size 64 --n_components 10 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.05  >> out_run09.log &
CUDA_VISIBLE_DEVICES=1 nohup python train_kronecker.py --batch_size 64 --n_components 7 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.05   >> out_run10.log &
CUDA_VISIBLE_DEVICES=2 nohup python train_kronecker.py --batch_size 64 --n_components 6 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.05   >> out_run11.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_kronecker.py --batch_size 64 --n_components 5 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.05   >> out_run12.log &

CUDA_VISIBLE_DEVICES=0 nohup python train_kronecker.py --batch_size 64 --n_components 4 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.05 >> out_run13.log &
CUDA_VISIBLE_DEVICES=1 nohup python train_kronecker.py --batch_size 64 --n_components 5 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.05  >> out_run14.log &
CUDA_VISIBLE_DEVICES=2 nohup python train_kronecker.py --batch_size 64 --n_components 3 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.05  >> out_run15.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_kronecker.py --batch_size 64 --n_components 1 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0.05  >> out_run16.log &

# CUDA_VISIBLE_DEVICES=3 nohup python train_multistep.py --batch_size 64 --n_components 1 --nhid 16  --reg_coef 0 --mse_coef 1 --rho 0 --diag  >> out_run17.log &
