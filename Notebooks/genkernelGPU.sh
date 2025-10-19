#!/bin/bash
#SBATCH --job-name=generateKernels        
#SBATCH --output=res_%j.txt
#SBATCH --error=err_%j.txt
#SBATCH --partition=submit-gpu
#SBATCH --mem-per-cpu=100000
#SBATCH --constraint=nvidia_a30
#SBATCH --gres=gpu:1

cd /work/submit/norwa667/workspaces/QuantumKernelEstimation/Notebooks
source ../quantum/bin/activate
echo "GPU Gen Kern"
echo "Job starting at $(date)"
python GenerateKernelsGPU.py
echo "Job finished at $(date)"

