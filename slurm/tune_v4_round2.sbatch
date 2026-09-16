#!/bin/bash
# =============================================================================
# v4 round 2. Round 1 showed two things:
#   * lower learning rate helps (lr5e-5 was the only config to improve BOTH
#     validation loss and taxonomy MSE), and it hit the 400-epoch cap still
#     improving -- so its result is an underestimate;
#   * the real mechanism is loss imbalance. v4 packs the same functional mass
#     into 195 pathways instead of cMD's 398, so the preprint's x10 factor makes
#     each pathway feature 23.7x a taxonomy feature (cMD: 5.1x). Pathways then
#     take 82-95% of the reconstruction loss (cMD: 42%) and taxonomy barely trains.
#
# Tasks 0-3 are a clean 2x2 factorial: {pathways-scale 100, 49} x {lr base, 5e-5}.
# pathways-scale 49 gives v4 the same per-feature pathway magnitude as cMD (2.513).
# Tasks 4-7 extend the two axes.
#
# Submit all:        sbatch slurm/tune_v4_round2.sbatch
# First wave only:   sbatch --array=0-3 slurm/tune_v4_round2.sbatch
# =============================================================================
#SBATCH --job-name=v4t2
#SBATCH --partition=gpu
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:55:00
#SBATCH --array=0-7
#SBATCH --output=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk/gut-former/logs/%x_%A_%a.out
#SBATCH --error=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk/gut-former/logs/%x_%A_%a.err

set -euo pipefail

BASE=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk
REPO="$BASE/gut-former"
ENV="$BASE/envs/gut-former"
DATA="$REPO/data"

PSCALE=100; LR=1.893292917167e-4; BS=16          # round-1 baseline

case "$SLURM_ARRAY_TASK_ID" in
  0)                              TAG=base ;;
  1) PSCALE=49;                   TAG=scale49 ;;
  2)             LR=5e-5;         TAG=lr5e-5 ;;
  3) PSCALE=49;  LR=5e-5;         TAG=scale49_lr5e-5 ;;
  4) PSCALE=21.4;                 TAG=scale21 ;;
  5) PSCALE=21.4; LR=5e-5;        TAG=scale21_lr5e-5 ;;
  6)             LR=2e-5;         TAG=lr2e-5 ;;
  7)             LR=5e-5; BS=64;  TAG=lr5e-5_bs64 ;;
  *) echo "Unexpected SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
RUN_NAME="v4t2_$(printf '%02d' "$SLURM_ARRAY_TASK_ID")_${TAG}"

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

echo "config: $RUN_NAME  pathways_scale=$PSCALE lr=$LR batch=$BS"

python src/gut_former/run_training.py \
  --taxonomy-file  "$DATA/taxonomy_v4.csv" \
  --pathways-file  "$DATA/pathways_v4.csv" \
  --split-file     "$DATA/splits.csv" \
  --split-column   tuning_split_random \
  --run-name       "$RUN_NAME" \
  --pathways-scale "$PSCALE" \
  --learning_rate  "$LR" \
  --batch_size     "$BS" \
  --epochs         1200 \
  --patience       100
