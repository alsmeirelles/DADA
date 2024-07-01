#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 25:00:00
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

cd $LOCAL/test/

echo 'Uncompressing data to LOCAL'

cp /pylon5/ac3uump/alsm/active-learning/data/lym_cnn_training_data.tar $LOCAL/test/
tar -xf lym_cnn_training_data.tar -C $LOCAL/test

cd /pylon5/ac3uump/alsm/active-learning/Segframe

echo '[VIRTUALENV]'
source /pylon5/ac3uump/alsm/venv/bin/activate

module load cuda/9.0

echo '[START] training'
date +"%D %T"

time python3 main.py -i -v --al -predst $LOCAL/test/lym_cnn_training_data/ -split 0.9 0.05 0.05 -net BayesVGG16 -data CellRep -init_train 2000 -dropout_steps 10 -ac_function random_sample -acquire 1000 -wpath random_weights -model_dir random_model -logdir random_logs -d -e 30 -b 96 -tdim 250 250 -out logs/ -cpu 9 -gpu 3 -tnorm -aug -tn

echo '[FINAL] done training'

deactivate 

date +"%D %T"


