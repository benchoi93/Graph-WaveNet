# CUDA_VISIBLE_DEVICES=0 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --batch_size 64 --n_components 1 --nhid 64  --rho 1 >> out_run02.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --batch_size 64 --n_components 3 --nhid 64  --rho 1 >> out_run03.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --batch_size 64 --n_components 5 --nhid 64  --rho 1 >> out_run04.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --batch_size 64 --n_components 10 --nhid 64  --rho 1 >> out_run05.log &

CUDA_VISIBLE_DEVICES=0 nohup wandb agent benchoi93/GWN/sxppna0d >> out_run01.log &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent benchoi93/GWN/sxppna0d >> out_run02.log &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent benchoi93/GWN/sxppna0d >> out_run03.log &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent benchoi93/GWN/sxppna0d >> out_run04.log &

