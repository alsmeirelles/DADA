#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 25:00:00
#SBATCH --gres=gpu:volta16:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:/pylon5/ac3uump/alsm/lib64/python3.6/site-packages:$PYTHONPATH

if [ ! -d $LOCAL/test ]
then
    mkdir $LOCAL/test;
fi


cd /pylon5/ac3uump/alsm/active-learning/Segframe

echo '[VIRTUALENV]'
source /pylon5/ac3uump/alsm/venv/bin/activate

module load cuda/9.0

echo '[START] training'
date +"%D %T" 

time python3 main.py -i -v --al -predst ~/.keras/datasets -split 0.857 0.013 0.13 -net GalKNet -data MNIST -bal -init_train 20 -ac_steps 50 -dropout_steps 100 -ac_function bayesian_varratios -acquire 20 -k -e 50 -b 128 -f1 0 -tnorm -out logs/ -cpu 9 -gpu 2 -tn -wpath results/MN-35 -model_dir results/MN-35 -logdir results/MN-35

echo '[FINAL] done training'

deactivate 

date +"%D %T"


