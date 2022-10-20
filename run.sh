# CUDA_VISIBLE_DEVICES=0 nohup python train_tensor.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 300 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 1 --learning_rate 0.001 --nonlinearity softplus  >> out_run01.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_tensor.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 300 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 1 --learning_rate 0.001 --nonlinearity softplus  >> out_run02.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_tensor.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 300 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 10 --learning_rate 0.001 --nonlinearity softplus  >> out_run03.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_tensor.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 300 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 100 --learning_rate 0.001 --nonlinearity softplus  >> out_run04.log &

# CUDA_VISIBLE_DEVICES=0 nohup python train_tensor.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 300 --nhid 32  --reg_coef 0.001 --mse_coef 1 --rho 0.1  --nonlinearity softplus  >> out_run01.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_tensor.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 200 --nhid 32  --reg_coef 0.001 --mse_coef 1 --rho 0.1  --nonlinearity softplus  >> out_run02.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_tensor.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 150 --nhid 32  --reg_coef 0.001 --mse_coef 1 --rho 0.1  --nonlinearity softplus  >> out_run03.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_tensor.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 100 --nhid 32  --reg_coef 0.001 --mse_coef 1 --rho 0.1  --nonlinearity softplus  >> out_run04.log &

# CUDA_VISIBLE_DEVICES=1 nohup python train_residual.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 5 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001  --nonlinearity softplus  >> out_run02.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_residual.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 10 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001  --nonlinearity softplus  >> out_run03.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_residual.py  --gcn_bool --adjtype doubletransition --addaptadj  --randomadj   --batch_size 64 --num-rank 20 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.001  --nonlinearity softplus  >> out_run04.log &


CUDA_VISIBLE_DEVICES=1 nohup python train_residual.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --batch_size 64 --num-rank 5 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.0001  --nonlinearity softplus  >> out_run02.log &
CUDA_VISIBLE_DEVICES=2 nohup python train_residual.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --batch_size 64 --num-rank 10 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.0001  --nonlinearity softplus  >> out_run03.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_residual.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --batch_size 64 --num-rank 20 --nhid 32  --reg_coef 0 --mse_coef 1 --rho 0.0001  --nonlinearity softplus  >> out_run04.log &

