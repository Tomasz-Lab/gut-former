#!/bin/bash
#SBATCH --job-name=FT_gutformer_cross
#SBATCH --partition=gpu
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk/gut-former/logs/%x_%A_%a.out
#SBATCH --error=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk/gut-former/logs/%x_%A_%a.err

set -euo pipefail

BASE=/cephfs/team/structural_and_functional_genomics/neural-net/piotr.urbanczyk
REPO="$BASE/gut-former"
ENV="$BASE/envs/gut-former"
DATA="$REPO/data"

case "$SLURM_ARRAY_TASK_ID" in
  0) TAXONOMY=taxonomy_cMD.csv; PATHWAYS=pathways_cMD.csv; SPLIT_COLUMN="cohort_split_age_category+disease";  RUN_NAME=cmd_cohort ;;
  1) TAXONOMY=taxonomy_cMD.csv; PATHWAYS=pathways_cMD.csv; SPLIT_COLUMN="random_split";                       RUN_NAME=cmd_random ;;
  2) TAXONOMY=taxonomy_v4.csv;  PATHWAYS=pathways_v4.csv;  SPLIT_COLUMN="cohort_split_age_category+disease";  RUN_NAME=v4_cohort  ;;
  3) TAXONOMY=taxonomy_v4.csv;  PATHWAYS=pathways_v4.csv;  SPLIT_COLUMN="random_split";                       RUN_NAME=v4_random  ;;
  *) echo "Unexpected SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

source /cephfs/software/miniforge/etc/profile.d/conda.sh
set +u
conda activate "$ENV"
set -u

cd "$REPO"
export PYTHONPATH="$REPO/src"
mkdir -p "$REPO/output"

# walkaround for not mounted $HOME
CACHE_ROOT="${SCRATCH_JOB:-${TMPDIR:-/tmp}}/gutformer_cache"
export HOME="$CACHE_ROOT"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/mpl"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

python src/gut_former/run_training.py \
  --taxonomy-file "$DATA/$TAXONOMY" \
  --pathways-file "$DATA/$PATHWAYS" \
  --split-file    "$DATA/splits.csv" \
  --split-column  "$SPLIT_COLUMN" \
  --run-name      "$RUN_NAME" \
  --epochs        1000
