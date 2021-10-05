#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:volta16:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

module load tensorflow/1.8_py3_gpu
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH

if [ ! -d $LOCAL/test ]
then
    mkdir $LOCAL/test;
fi

cd $LOCAL/test/

echo 'Uncompressing data to LOCAL'

cp /pylon5/ac3uump/alsm/active-learning/data/lym_cnn_training_data.tar $LOCAL/test/
tar -xf lym_cnn_training_data.tar -C $LOCAL/test

cd /pylon5/ac3uump/alsm/active-learning/Segframe

echo '[START] training'
date +%s

time python3 main.py -i -v --train -predst $LOCAL/test/lym_cnn_training_data/ -split 0.9 0.05 0.05 -net VGG16A3 -data CellRep -e 25 -b 128 -tdim 250 250 -tnorm -out logs/ -cpu 8 -gpu 2 --pred 
echo '[FINAL] done training'
date +%s

#cp $LOCAL/test/weights/*.hdf5 full-test-f5v2/
#cp $LOCAL/test/weights/*.h5 full-test-f5v2/
#cp $LOCAL/test/*.txt full-test-f5v2/
#cp $LOCAL/test/*.pik full-test-f5v2/
#cp $LOCAL/test/*.csv batch_result/

