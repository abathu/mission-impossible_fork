#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=tc067-s2678328
#SBATCH --time=14:00:00
#SBATCH --output=tag_russian_%j.out
#SBATCH --error=tag_russian_%j.err

source /work/tc067/tc067/s2678328/myvenv/bin/activate
export TRANSFORMERS_CACHE=/work/tc067/tc067/s2678328/hf_cache

cd /work/tc067/tc067/s2678328/mission-impossible/data

python tag_rus.py /work/tc067/tc067/s2678328/babyLM/Russain/translated_file/Russian/*/*.txt