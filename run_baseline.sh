#!/bin/bash
#PBS -N edgeface_baseline_run
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64g
#PBS -l walltime=01:00:00
#PBS -j oe

# Navigate to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Activate the renamed conda environment
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate GPU-env

echo "=========================================================="
echo "Starting EdgeFace Baseline Evaluation..."
echo "=========================================================="
echo "Evaluating: edgeface_s_gamma_05"
python eval_edgeface.py edgeface_s_gamma_05

echo ""
echo "Evaluating: edgeface_xs_gamma_06"
python eval_edgeface.py edgeface_xs_gamma_06

echo ""
echo "=========================================================="
echo "Starting EdgeFace Baseline Synthetic Training..."
echo "=========================================================="
# Run the PyTorch distributed training script with 1 GPU using the synthetic dataset
python -m torch.distributed.run --nproc_per_node=1 train_v2.py configs/baseline_synthetic.py

echo ""
echo "=========================================================="
echo "Baseline Run Completed."
echo "=========================================================="
