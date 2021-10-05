#!/bin/bash
#SBATCH -p GPU
#SBATCH -t 40:00:00
#SBATCH --gres=gpu:p100:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

DIRID="EN-23"
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:/pylon5/ac3uump/alsm/lib64/python3.6/site-packages:$PYTHONPATH

if [ ! -d $LOCAL/test ]
then
    mkdir $LOCAL/test;
fi

cd $LOCAL/test/

#echo 'Uncompressing data to LOCAL'

#cp -r /pylon5/ac3uump/alsm/active-learning/data/extracted $LOCAL/test/
#tar -xf lym_cnn_training_data.tar -C $LOCAL/test

cd /pylon5/ac3uump/alsm/active-learning/Segframe

echo '[VIRTUALENV]'
source /pylon5/ac3uump/alsm/venv/bin/activate

module load cuda/9.0

echo '[START] training'
date +"%D %T"

time python3 main.py -i -v --al -strategy EnsembleTrainer -predst /pylon5/ac3uump/alsm/active-learning/data/nds300 -split 0.90 0.01 0.09 -net Inception -data CellRep -init_train 500 -ac_steps 20  -emodels 3 -ac_function km_uncert -un_function ensemble_bald -acquire 200 -d -e 50 -b 60 -tdim 240 240 -clusters 20 -f1 25 -out logs/ -cpu 5 -gpu 2 -tn -sv -nsw -wpath results/$DIRID -model_dir results/$DIRID -logdir results/$DIRID -cache results/$DIRID -pca 50 -load_train 

echo '[FINAL] done training'

deactivate 

date +"%D %T"


