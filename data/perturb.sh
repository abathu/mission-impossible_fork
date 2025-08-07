#!/bin/bash
#perturb.sh

#SBATCH --job-name=perturb_test
#SBATCH --output=/home/s2678328/mission-impossible-language-models/logs/slurm-%j.out
#SBATCH --error=/home/s2678328/mission-impossible-language-models/logs/slurm-%j.out


# ------- THis is for tagging the data -------
# $1 is the first argument passed to the script

# echo "$1"


# python3 /home/s2678328/mission-impossible-language-models/data/tag.py "$1"  #/*.txt

# echo "Done"




# ------- This is for perturbing the data -------

cd /home/s2678328/mission-impossible-language-models/data


echo "-------------------------------------------------------------------------------"
echo "Arguments"
echo "-------------------------------------------------------------------------------"

echo "Perturbation type: $1"
echo "Train set: $2"


# Create perturbed dataset for all splits
echo " -------------------------------------------------------------------------------"
echo "Creating perturbed dataset for all splits"
echo "------------------------------------------------------------------------------- "

cd ../data

echo "python3 perturb.py $1 $2"
python3 perturb.py $1 $2
echo "python3 perturb.py $1 dev"
python3 perturb.py $1 dev



# echo "
# python3 perturb.py $1 test"
# python3 perturb.py $1 test
# echo "
# python3 perturb.py $1 unittest"
# python3 perturb.py $1 unittest

# cd ..




echo "Done"