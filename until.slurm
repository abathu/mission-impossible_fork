#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=tc067-s2678328
#SBATCH --time=14:00:00

source /work/tc067/tc067/s2678328/myvenv/bin/activate
export TRANSFORMERS_CACHE=/work/tc067/tc067/s2678328/hf_cache
export STANZA_RESOURCES_DIR=/work/tc067/tc067/s2678328/stanza_resources


cd /work/tc067/tc067/s2678328/mission-impossible/data

srun python tag_rus.py /work/tc067/tc067/s2678328/babyLM/Russain/translated_file/Russain #/work/tc067/tc067/s2678328/babyLM/Russian/translated_file/Russian/
#
#python tag_rus.py /work/tc067/tc067/s2678328/babyLM/Russain/translated_file/Russian/

#python tag_rus.py /work/tc067/tc067/s2678328/babyLM/Russain/translated_file/Russian/*/*.txt


echo "done script run finnished  "
