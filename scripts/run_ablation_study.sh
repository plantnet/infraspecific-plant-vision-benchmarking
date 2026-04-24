#!/bin/bash

# --- 1. CONFIGURATION ---
DATASET_NAME="bananas" # Change to your target dataset
ROOT_IMG="./data/$DATASET_NAME/images"
META_TRAIN="./data/$DATASET_NAME/train.csv"
META_TEST="./data/$DATASET_NAME/test.csv"
ORGAN_COL="organ" # Name of the column in your CSV

# --- 2. SELECT THE CORE MODEL FOR ABLATION ---
# Usually, we perform ablation on the best performing model (e.g., DinoV2)
MODEL="vit_base_patch14_dinov2.lvd142m"

# --- 3. AUTOMATIC ORGAN DETECTION ---
# We extract unique organ names from the CSV column (excluding the header)
ORGANS=$(tail -n +2 "$META_TRAIN" | cut -d',' -f4 | sort | uniq) # Assumes 'organ' is the 4th column

echo "Starting Ablation Study for $DATASET_NAME using $MODEL"
echo "Organs to test: $ORGANS"

# --- 4. EXECUTION LOOP ---
for EXCLUDE in $ORGANS
do
    echo "------------------------------------------------"
    echo "RUNNING EXPERIMENT: Excluding organ '$EXCLUDE'"
    echo "------------------------------------------------"

    # Define a specific output folder for this ablation
    ABLATION_DIR="./results/$DATASET_NAME/ablation_excl_$EXCLUDE"
    mkdir -p "$ABLATION_DIR"

    # Train while excluding one organ
    python train.py \
        --model "$MODEL" \
        --root-dir "$ROOT_IMG" \
        --metadata-train "$META_TRAIN" \
        --metadata-val "$META_TRAIN" \
        --exclude-organ "$EXCLUDE" \
        --epochs 100 \
        --frozen \
        --lr 0.001

    # Validate to see the impact on accuracy
    python validate.py \
        --model "$MODEL" \
        --checkpoint "./output/model_best.pth" \
        --root-dir "$ROOT_IMG" \
        --metadata-test "$META_TEST" \
        --results-dir "$ABLATION_DIR"
done

echo "Ablation study complete. Results are stored in ./results/$DATASET_NAME/"
