
# CUDA_VISIBLE_DEVICES=0 nohup python train.py --delay 1 --batch_size 128 --fix_L_space --fix_L_batch --rho 0 >> out1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --delay 1 --batch_size 128 --fix_L_space --fix_L_batch --rho 1 >> out1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --delay 4 --batch_size 32 --fix_L_space  --rho 1 >> out2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --delay 4 --batch_size 32  --rho 1  >> out3.log 2>&1 &