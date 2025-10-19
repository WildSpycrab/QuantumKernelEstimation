#!/bin/bash
#SBATCH --job-name=generateKernels        
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --partition=submit
#SBATCH --mem-per-cpu=10000

cd /work/submit/norwa667/workspaces/QuantumKernelEstimation/Notebooks
source ../quantum/bin/activate
echo "cpu kern"
echo "Job starting at $(date)"
python GenerateKernels.py
echo "Job finished at $(date)"

