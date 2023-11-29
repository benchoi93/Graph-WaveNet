
CUDA_VISIBLE_DEVICES=0 nohup wandb agent benchoi93/GWN_1120/9lya3k7v >> out_run00.log &
CUDA_VISIBLE_DEVICES=1 nohup wandb agent benchoi93/GWN_1120/9lya3k7v >> out_run01.log &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent benchoi93/GWN_1120/9lya3k7v >> out_run02.log &