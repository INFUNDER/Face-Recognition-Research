#!/bin/bash
#PBS -N edgeface_synthetic_baseline
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64g
#PBS -l walltime=01:00:00
#PBS -j oe

# Navigate to the directory where the job was submitted
cd $PBS_O_WORKDIR

source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate GPU-env

# We do not load the system CUDA module because it conflicts with PyTorch's bundled CUDA libraries.
# Let PBS handle GPU assignment automatically

echo "Starting EdgeFace Synthetic Baseline Training..."

# Run the PyTorch distributed training script with 1 GPU
python -m torch.distributed.run --nproc_per_node=1 train_v2.py configs/baseline_synthetic.py

echo "Training Completed."
