#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --exclude=v034
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alsmeirelles@gmail.com

#echo commands to stdout
set -x

DIRID="JS/JS-6"

cd /ocean/projects/asc130006p/alsm/active-learning/Segframe

echo '[VIRTUALENV]'
source /ocean/projects/asc130006p/alsm/venv/bin/activate

#Load CUDA and set LD_LIBRARY_PATH
module load cuda/10.0.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ocean/projects/asc130006p/alsm/venv/lib64/cuda-10.0.0

echo '[START] training'
date +"%D %T"

time python3 Utils/JensenShannon.py -i -v --train -nets Xception -tnet EFInception -data CellRep -predst /ocean/projects/asc130006p/alsm/active-learning/data/nds300 -e 50 -train_set 4000 -val_set 100 -un_set 2000 -tnphi 1 -tdim 240 240 -ac_function bayesian_bald -strategy ActiveLearningTrainer -dropout_steps 20 -emodels 3 -out logs -lr 0.0001 -logdir results/$DIRID -wpath results/$DIRID -model_dir results/$DIRID -cache cache/JS -save_dt -k -d -b 64 -gpu 1 -cpu 15 -acquire 100 -phis 1

#-plw -lyf 103 

#-tnet EFInception -tnpred 2 -tnphi 2 -f1 30 -lr 0.0001 

#-wsilist TCGA-BL-A13J-01Z-00 TCGA-FR-A728-01Z-00 TCGA-EE-A2MH-01Z-00 TCGA-C5-A1MH-01Z-00 TCGA-US-A77G-01Z-00 -wsimax 1.0 1.0 1.0 1.0 0.5

echo '[FINAL] done training'

deactivate 

date +"%D %T"


