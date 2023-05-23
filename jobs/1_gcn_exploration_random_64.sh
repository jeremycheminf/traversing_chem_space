#!/bin/bash
#SBATCH --job-name=1_gcn_exploration_random_64
#SBATCH --output=/home/tilborgd/projects/Active_Learning_Simulation/out/1_gcn_exploration_random_64.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=120:00:00

source $HOME/anaconda3/etc/profile.d/conda.sh
export PYTHONPATH="$PYTHONPATH:$HOME/projects/Active_Learning_Simulation"
$HOME/anaconda3/envs/molml/bin/python -u $HOME/projects/Active_Learning_Simulation/experiments/main.py -o /home/tilborgd/projects/Active_Learning_Simulation/results -acq exploration -bias random -arch gcn -batch_size 64 > $HOME/projects/Active_Learning_Simulation/results/gcn_exploration_random_64.log