#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:0:0    
#SBATCH --mail-user=<seongjin.choi@mcgill.ca>
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:1

cd ~/$projects/Graph-WaveNet/
module purge
module load python/3.7.9 scipy-stack
source ~/py37/bin/activate
pip install -r requirements.txt

python train_resmix.py
