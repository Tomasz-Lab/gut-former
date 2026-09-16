#!/bin/bash
# =============================================================================
# v4 hyperparameter probe: one-factor-at-a-time from the preprint's cMD-tuned
# configuration, to test whether those hyperparameters transfer to the 462-feature
# v4 taxonomy space.
#
# Selection uses `tuning_split_random`, whose validation rows are carved from the
# TRAIN half of random_split. The 1505 real test samples are labelled "holdout"
# and are silently excluded by train_val_split -- so tuning never sees them and
# the downstream benchmark stays uncontaminated.
#
# Resources are sized from sacct on job 8978: MaxRSS was 1.25G and the GPU does
# the work, so 16G/4cpu is ~12x margin. Requesting the training script's 140G/12cpu
# for 8 concurrent tasks reserves the whole node and blocks scheduling for days.
# Measured 64 s/epoch for v4, so 400 epochs ~= 7.1h (emb256 ~= 14h).
#
# Submit: sbatch slurm/tune_v4.sbatch
# =============================================================================
#SBATCH --job-name=v4tune
#SBATCH --partition=gpu
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:55:00
#SBATCH --array=0-7
#SBATCH --output=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk/gut-former/logs/%x_%A_%a.out
#SBATCH --error=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk/gut-former/logs/%x_%A_%a.err

set -euo pipefail

BASE=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk
REPO="$BASE/gut-former"
ENV="$BASE/envs/gut-former"
DATA="$REPO/data"

# baseline = the preprint's configuration (what v4_random already used)
EMB=128; LAT=64; LR=1.893292917167e-4; BS=16

case "$SLURM_ARRAY_TASK_ID" in
  0) TAG=base ;;
  1) LAT=128;  TAG=lat128 ;;
  2) LAT=256;  TAG=lat256 ;;
  3) LR=5e-4;  TAG=lr5e-4 ;;
  4) LR=5e-5;  TAG=lr5e-5 ;;
  5) EMB=256;  TAG=emb256 ;;
  6) BS=64;    TAG=bs64 ;;
  7) LAT=128; LR=5e-4; TAG=lat128_lr5e-4 ;;
  *) echo "Unexpected SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
RUN_NAME="v4tune_$(printf '%02d' "$SLURM_ARRAY_TASK_ID")_${TAG}"

source /cephfs/software/miniforge/etc/profile.d/conda.sh
set +u
conda activate "$ENV"
set -u

cd "$REPO"
export PYTHONPATH="$REPO/src"
mkdir -p "$REPO/output"

CACHE_ROOT="${SCRATCH_JOB:-${TMPDIR:-/tmp}}/gutformer_cache"
export HOME="$CACHE_ROOT"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/mpl"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

echo "config: $RUN_NAME  embedding=$EMB latent=$LAT lr=$LR batch=$BS"

python src/gut_former/run_training.py \
  --taxonomy-file "$DATA/taxonomy_v4.csv" \
  --pathways-file "$DATA/pathways_v4.csv" \
  --split-file    "$DATA/splits.csv" \
  --split-column  tuning_split_random \
  --run-name      "$RUN_NAME" \
  --embedding_dim "$EMB" \
  --latent_dim    "$LAT" \
  --learning_rate "$LR" \
  --batch_size    "$BS" \
  --epochs        400 \
  --patience      60
