# CUDA_VISIBLE_DEVICES=0 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --n_components 5 --nhid 64  --rho 0.001 --mix_mean "True" >> out_run02.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --n_components 3 --nhid 64  --rho 0.001 --mix_mean "True" >> out_run02.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --n_components 3 --nhid 64  --rho 1 --mix_mean "True" >> out_run02.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --n_components 5 --nhid 64  --rho 0 --mix_mean "True" >> out_run02.log &

CUDA_VISIBLE_DEVICES=0 nohup wandb agent benchoi93/GWN/pwum6t5p >> out_run01.log &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent benchoi93/GWN/pwum6t5p >> out_run02.log &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent benchoi93/GWN/pwum6t5p >> out_run03.log &
CUDA_VISIBLE_DEVICES=3 nohup wandb agent benchoi93/GWN/pwum6t5p >> out_run04.log &

# CUDA_VISIBLE_DEVICES=0 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epoch 200 --n_components 1 --nhid 64  --rho 0 --diag --mix_mean "True" >> out_run01.log &
# CUDA_VISIBLE_DEVICES=1 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epoch 200 --n_components 1 --nhid 64  --rho 0 --diag --mix_mean "False" >> out_run02.log &
# CUDA_VISIBLE_DEVICES=2 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epoch 200 --n_components 3 --nhid 64  --rho 0.001 --mix_mean "False" >> out_run03.log &
# CUDA_VISIBLE_DEVICES=3 nohup python train_new.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --epoch 200 --n_components 3 --nhid 64  --rho 0.001 --mix_mean "True" >> out_run04.log &
