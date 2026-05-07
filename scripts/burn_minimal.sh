#!/bin/bash
#SBATCH --job-name=envisage_burn
#SBATCH --account=p_meiler_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_burn_%A_%a.out
#SBATCH --array=0-2

# Burn-it-down minimal inference. One task per procedure.
# array=0 → bleph, 1 → rhino, 2 → rhytid. Run bleph first by submitting
# with --array=0 explicitly.

set -euo pipefail

PROCEDURES=("blepharoplasty" "rhinoplasty" "rhytidectomy")
PROCEDURE=${PROCEDURES[$SLURM_ARRAY_TASK_ID]}

echo "=== Envisage BURN minimal: $PROCEDURE ==="
echo "Job: $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
date

source ~/miniconda3/etc/profile.d/conda.sh
conda activate landmarkdiff

export HF_HOME=/data/p_csb_meiler/agarwm5/.cache/huggingface
export HF_TOKEN=$(cat /data/p_csb_meiler/agarwm5/infinity/envisage/.env | grep HF_TOKEN | cut -d= -f2)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORK_DIR=/data/p_csb_meiler/agarwm5/infinity/envisage
cd "$WORK_DIR"

TEST_SPLIT=/data/p_csb_meiler/agarwm5/landmarkdiff_work/LandmarkDiff/data/hda_splits/test
OUTPUT_DIR="evaluation/burn/${PROCEDURE}"

# Cases to run: bleph=10 cases, rhino=10, rhytid=9 (all it has)
MAX_CASES=10
case "$PROCEDURE" in
  blepharoplasty)
    # Start with the M1/M2 "win" cases so we can compare direct to v2
    CASES="Eyelid_56 Eyebrow_105 Eyebrow_107 Eyebrow_13 Eyelid_74 Eyelid_98 Eyelid_93 Eyelid_91 Eyebrow_28 Eyebrow_50"
    ;;
  rhinoplasty)
    # Nose_27 is the historical demo case
    CASES="Nose_27 Nose_30 Nose_31 Nose_33 Nose_50 Nose_51 Nose_57 Nose_102 Nose_113 Nose_116"
    ;;
  rhytidectomy)
    # Skip Facelift_08 (HDA pairing bug); use other cases
    CASES=""
    MAX_CASES=9
    ;;
esac

if [[ -n "$CASES" ]]; then
  python3 -m scripts.burn_minimal \
      --test-split "$TEST_SPLIT" \
      --output-dir "$OUTPUT_DIR" \
      --procedure "$PROCEDURE" \
      --cases $CASES \
      --strength 0.50 \
      --resolution 1024 \
      --steps 30 \
      2>&1
else
  python3 -m scripts.burn_minimal \
      --test-split "$TEST_SPLIT" \
      --output-dir "$OUTPUT_DIR" \
      --procedure "$PROCEDURE" \
      --max-cases $MAX_CASES \
      --strength 0.50 \
      --resolution 1024 \
      --steps 30 \
      2>&1
fi

echo "=== Done: $PROCEDURE ==="
date
