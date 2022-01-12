#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 30:00:00
#SBATCH --gpus=v100-32:2
#SBATCH --exclude=v034
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
#set -x

DIRID="DB/DB-43"
cd /ocean/projects/asc130006p/alsm/active-learning/Segframe

echo '[VIRTUALENV]'
source /ocean/projects/asc130006p/alsm/venv/bin/activate

#Load CUDA and set LD_LIBRARY_PATH
module load cuda/10.0.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ocean/projects/asc130006p/alsm/venv/lib64/cuda-10.0.0

echo '[START] training'
date +"%D %T"

time python3 main.py -i -v --train --pred -predst /ocean/projects/asc130006p/alsm/active-learning/data/AL-214-Aug -split 0.9 0.005 0.0 -net EFInception -data AqSet -d -k -e 50 -b 128 -tdim 240 240 -f1 10 -out logs/ -cpu 15 -gpu 2 -tn -sv -nsw -wpath results/$DIRID -model_dir results/$DIRID -logdir results/$DIRID -cache results/$DIRID -test_dir /ocean/projects/asc130006p/alsm/active-learning/data/T5-2-test -phi 1 


echo '[FINAL] done training'

deactivate 

date +"%D %T"


