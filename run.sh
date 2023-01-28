CUDA_VISIBLE_DEVICES=0 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 1 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0 --loss mse  >> out_run01.log &

CUDA_VISIBLE_DEVICES=0 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 1 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0.001 --loss mse >> out_run02.log &
CUDA_VISIBLE_DEVICES=1 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 2 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0.001 --loss mse >> out_run03.log &
CUDA_VISIBLE_DEVICES=2 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 3 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0.001 --loss mse >> out_run04.log &
CUDA_VISIBLE_DEVICES=3 nohup python train_resmix.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epochs 500 --batch_size 64 --num-rank 5 --nhid 64  --reg_coef 0 --mse_coef 1 --rho 0.001 --loss mse >> out_run05.log &

