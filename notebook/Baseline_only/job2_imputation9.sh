#!/bin/bash
#SBATCH --job-name=missing_imputation_test    # Job name
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10                 # Number of CPU cores increased from 5->10
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-0:0:0    
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=yjkweon24@berkeley.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=facte_%x_%j.out        # Output file (%j is job ID)
#SBATCH --error=facte_%x_%j.err         # Error file
#SBATCH --account=def-rtc

module load cuda/12.2

# Activate your virtual environment (if you have one)
source ~/jinenv/bin/activate

# Change to project directory
cd /home/yjkweon2/projects/def-rtc/yjkweon2/Missing_imputation_local/Codes/Baseline_only

python test2_imputation9.py
