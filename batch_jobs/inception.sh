#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 44:00:00
#SBATCH --gres=gpu:volta16:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

DIRID="DB-16"
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

time python3 main.py -i -v --train --pred -predst /pylon5/ac3uump/alsm/active-learning/data/nds300 -split 0.9 0.01 0.09 -net Inception -data CellRep -d -e 20 -b 62 -tdim 240 240 -f1 10 -out logs/ -cpu 9 -gpu 2 -tn -sv -nsw -wpath results/$DIRID -model_dir results/$DIRID -logdir results/$DIRID -cache results/$DIRID -wsi_split 5 -pred_size 15000 


echo '[FINAL] done training'

deactivate 

date +"%D %T"


