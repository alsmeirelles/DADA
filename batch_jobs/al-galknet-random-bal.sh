#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 15:00:00
#SBATCH --gres=gpu:volta16:3
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

time python3 main.py -i -v --al -predst ~/.keras/datasets -split 0.857 0.013 0.13 -net GalKNet -data MNIST -bal -init_train 20 -ac_steps 50 -dropout_steps 100 -ac_function random_sample -acquire 20 -k -e 50 -b 96 -f1 0 -tnorm -out logs/ -cpu 9 -gpu 3 -tn -wpath results/MN-32 -model_dir results/MN-32 -logdir results/MN-32

echo '[FINAL] done training'

deactivate 

date +"%D %T"


