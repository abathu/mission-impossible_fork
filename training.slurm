#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --account=tc067-s2678328



# activate python env 
source /work/tc067/tc067/s2678328/mistral_env/bin/activate





echo "Current datetime: $(date +"%F %T")"

cd /work/tc067/tc067/s2678328/mission-impossible

CUDA_VISIBLE_DEVICES=0 python3 train.py --config conf/train_reverse_partial_100M_randinit_seed41.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.warmup_steps 300   --training_arguments.num_train_epochs 10


echo "Current datetime: $(date +"%F %T")"
