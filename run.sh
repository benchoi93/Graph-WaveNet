# CUDA_VISIBLE_DEVICES=0 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 2 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0 --nonlinearity softplus  >> out_run01.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 2 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.0001  --nonlinearity softplus  >> out_run02.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 2 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001  --nonlinearity softplus  >> out_run03.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 2 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.01  --nonlinearity softplus  >> out_run04.log &

# CUDA_VISIBLE_DEVICES=0 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 3 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0 --nonlinearity softplus  >> out_run11.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 3 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.0001  --nonlinearity softplus  >> out_run12.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 3 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001  --nonlinearity softplus  >> out_run13.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 3 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.01  --nonlinearity softplus  >> out_run14.log &

# CUDA_VISIBLE_DEVICES=0 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 4 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0 --nonlinearity softplus  >> out_run21.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 4 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.0001  --nonlinearity softplus  >> out_run22.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 4 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001  --nonlinearity softplus  >> out_run23.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 1000 --batch_size 64 --num-rank 4 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.01  --nonlinearity softplus  >> out_run24.log &


CUDA_VISIBLE_DEVICES=2 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 5 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0.01 --loss mae >> out_run01.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 5 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0.001 --loss mae  >> out_run02.log &
CUDA_VISIBLE_DEVICES=2 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 5 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0.01 --loss mse  >> out_run03.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 5 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0.001 --loss mse  >> out_run04.log &

# CUDA_VISIBLE_DEVICES=0 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 3 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0 --nonlinearity softplus  >> out_run05.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 3 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001 --nonlinearity softplus  >> out_run06.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 4 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0 --nonlinearity softplus  >> out_run07.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 4 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001 --nonlinearity softplus  >> out_run08.log &

# CUDA_VISIBLE_DEVICES=0 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 5 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0 --nonlinearity softplus  >> out_run09.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 5 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001 --nonlinearity softplus  >> out_run10.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 6 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0 --nonlinearity softplus  >> out_run11.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 6 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001 --nonlinearity softplus  >> out_run12.log &

