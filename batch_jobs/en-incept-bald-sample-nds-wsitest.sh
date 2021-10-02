#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 30:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

DIRID="DB-24"
cd /ocean/projects/asc130006p/alsm/active-learning/Segframe

echo '[VIRTUALENV]'
source /ocean/projects/asc130006p/alsm/venv/bin/activate

#Load CUDA and set LD_LIBRARY_PATH
module load cuda/10.0.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ocean/projects/asc130006p/alsm/venv/lib64/cuda-10.0.0

echo '[START] training'
date +"%D %T"

time python3 main.py -i -v --al -strategy EnsembleTrainer -predst /ocean/projects/asc130006p/alsm/active-learning/data/nds300 -split 0.90 0.01 0.09 -net EFInception -data CellRep -init_train 500 -ac_steps 2 -emodels 3 -ac_function ensemble_bald -acquire 200 -d -e 50 -b 62 -tdim 240 240 -f1 30 -out logs/ -cpu 6 -gpu 1 -tn -sv -nsw -wpath results/$DIRID -model_dir results/$DIRID -logdir results/$DIRID -cache results/$DIRID -sample 100000 -load_train -wsi_split 5 -pred_size 15000 -phi 2 -tnphi 2 -lr 0.0001 -k

#-tnet EFInception -tnpred 2

echo '[FINAL] done training'

deactivate 

date +"%D %T"


