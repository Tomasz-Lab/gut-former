#!/bin/bash
#SBATCH --job-name=gutformer_cross
#SBATCH --partition=gpu
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=/cephfs/team/structural_and_functional_genomics/neural-net/gut-former/logs/%x_%A_%a.out
#SBATCH --error=/cephfs/team/structural_and_functional_genomics/neural-net/gut-former/logs/%x_%A_%a.err

set -euo pipefail

REPO=/cephfs/team/structural_and_functional_genomics/neural-net/gut-former
DATA="$REPO/data"

case "$SLURM_ARRAY_TASK_ID" in
  0) TAXONOMY=taxonomy_cMD.csv; PATHWAYS=pathways_cMD.csv; SPLIT_COLUMN="cohort_split_age_disease"; RUN_NAME=cmd_cohort ;;
  1) TAXONOMY=taxonomy_cMD.csv; PATHWAYS=pathways_cMD.csv; SPLIT_COLUMN="random_split";             RUN_NAME=cmd_random ;;
  2) TAXONOMY=taxonomy_v4.csv;  PATHWAYS=pathways_v4.csv;  SPLIT_COLUMN="cohort_split_age_disease"; RUN_NAME=v4_cohort  ;;
  3) TAXONOMY=taxonomy_v4.csv;  PATHWAYS=pathways_v4.csv;  SPLIT_COLUMN="random_split";             RUN_NAME=v4_random  ;;
  *) echo "Unexpected SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

source "$REPO/.venv/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src"

python src/gut_former/run_training.py \
  --taxonomy-file "$DATA/$TAXONOMY" \
  --pathways-file "$DATA/$PATHWAYS" \
  --split-file    "$DATA/splits.csv" \
  --split-column  "$SPLIT_COLUMN" \
  --run-name      "$RUN_NAME" \
  --epochs        1000
