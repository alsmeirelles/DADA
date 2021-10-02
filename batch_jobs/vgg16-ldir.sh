#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 35:00:00
#SBATCH --gres=gpu:volta16:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

module load tensorflow/1.8_py3_gpu
export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:$PYTHONPATH

cd /pylon5/ac3uump/alsm/active-learning/Segframe

echo '[START] training'
date +%s

time python3 main.py -i -v --train -predst /pylon5/ac3uump/lhou/patches_train -split 0.94 0.01 0.05 -net VGG16 -data LDir -e 30 -b 128 -tdim 300 300 -out logs/ -cpu 12 -gpu 4 --pred -tn
echo '[FINAL] done training'
date +%s

#cp $LOCAL/test/weights/*.hdf5 full-test-f5v2/
#cp $LOCAL/test/weights/*.h5 full-test-f5v2/
#cp $LOCAL/test/*.txt full-test-f5v2/
#cp $LOCAL/test/*.pik full-test-f5v2/
#cp $LOCAL/test/*.csv batch_result/

