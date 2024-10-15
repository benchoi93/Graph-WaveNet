# CUDA_VISIBLE_DEVICES=0 nohup python train_new.py --gcn_bool --addaptadj  --randomadj --n_components 1 --rho 0  >> out_run00.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epoch 200 --n_components 5 --nhid 64  --rho 1  >> out_run04.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epoch 200 --n_components 10 --nhid 64  --rho 1  >> out_run04.log &


CUDA_VISIBLE_DEVICES=0 nohup wandb agent benchoi93/GWN_1120/9lya3k7v >> out_run00.log &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent benchoi93/GWN_1120/9lya3k7v >> out_run01.log &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent benchoi93/GWN_1120/9lya3k7v >> out_run02.log &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent benchoi93/GWN_1120/9lya3k7v >> out_run03.log &