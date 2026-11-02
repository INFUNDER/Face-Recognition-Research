#!/bin/bash
# Unattended HFR pipeline for the login node (needs internet, so not a PBS job):
# download -> align -> zero-shot baselines (fixed split) -> submit the 5-fold cross-validation.
# Every step is resumable, so the script can simply be restarted if it dies.
set -e
cd "$(dirname "$0")/../.."
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate GPU-env

echo "[$(date)] download"
python scripts/hfr/download_speakingfaces.py --workers 12

echo "[$(date)] align"
python scripts/hfr/align_speakingfaces.py --device cpu

echo "[$(date)] zero-shot baselines"
mkdir -p results/hfr/preliminary
for NET in edgeface_xs_gamma_06 edgeface_s_gamma_05; do
    python scripts/hfr/eval_hfr.py --network $NET --weight checkpoints/$NET.pt --out results/hfr/preliminary/zeroshot_${NET}_fp32.json
    python scripts/hfr/eval_hfr.py --network $NET --weight checkpoints/$NET.pt --int8 --out results/hfr/preliminary/zeroshot_${NET}_int8.json
done

echo "[$(date)] submit 5-fold cross-validation (the gpu queue runs at most 2 jobs per user)"
mkdir -p logs/hfr
for SHARDS in "0/8 2/8 4/8 6/8" "1/8 3/8 5/8 7/8"; do
    qsub -v CONFIG=scripts/hfr/configs/cv.txt,OUT_ROOT=$PWD/results/hfr/cv,SHARDS="$SHARDS" \
        -o $PWD/logs/hfr/cv_pbs.log scripts/hfr/run_experiments_multi.pbs
done
echo "afterwards: qsub scripts/hfr/eval_mixed_precision.pbs"
echo "[$(date)] pipeline done"
