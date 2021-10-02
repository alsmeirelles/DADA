#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:volta16:1
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

time python3 main.py -i -v --al -predst ~/.keras/datasets -split 0.857 0.013 0.13 -net BayesKNet -data MNIST -tnorm -init_train 20 -ac_steps 50 -dropout_steps 100 -ac_function bayesian_bald -acquire 20 -k -e 50 -b 128 -out logs/ -cpu 4 -gpu 1 -f1 0 -tn -wpath results/MN-59 -model_dir results/MN-59 -logdir results/MN-59

echo '[FINAL] done training'

deactivate 

date +"%D %T"


