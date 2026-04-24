#!/bin/bash

# ==========================================================
# 1. STORAGE CONFIGURATION
# ==========================================================
DATASET="plant-market-garden" 

# Chemins locaux flandre
DATA_ROOT="/fastdata/datasets/tibi/zenodo"
RESULT_ROOT="/data/infraspecies/vegetables/check_package_for_article/results"

DATASET_ROOT="$DATA_ROOT/$DATASET"
OUTPUT_BASE="$RESULT_ROOT/$DATASET"
IMG_PARENT_DIR="$DATASET_ROOT/images"
METADATA_TRAIN="$DATASET_ROOT/train.csv"
METADATA_TEST="$DATASET_ROOT/test.csv"

LABEL_COL="class_name" 
WORKERS=8 

# ==========================================================
# 2. EXPERIMENT DEFINITIONS
# Format: "MODEL : RES : CHECKPOINT : MODE (frozen/full)"
# ==========================================================
EXPERIMENTS=(
    # --- DINOv2 Base ---
    "vit_base_patch14_dinov2.lvd142m:518::frozen"
    "vit_base_patch14_dinov2.lvd142m:518::full"
    
    # --- DINOv2 Reg4 (PlantNet Pretrained) ---
    "vit_base_patch14_reg4_dinov2.lvd142m:518:/data/PlantNet/models/dinov2july2024/model_best_no_head.pth.tar:frozen"
    "vit_base_patch14_reg4_dinov2.lvd142m:518:/data/PlantNet/models/dinov2july2024/model_best_no_head.pth.tar:full"
    
    # --- ResNet50 ---
    "resnet50.a1_in1k:224::frozen"
    "resnet50.a1_in1k:224::full"
    
    # --- ConvNeXt ---
    "convnext_base.fb_in22k_ft_in1k_384:384::frozen"
    "convnext_base.fb_in22k_ft_in1k_384:384::full"
)

# ==========================================================
# 3. EXECUTION LOOP
# ==========================================================
for EXP in "${EXPERIMENTS[@]}"
do
    # Découpage des 4 champs
    IFS=":" read -r MODEL RES CKPT MODE <<< "$EXP"
    
    # Logique de nommage et flags
    FROZEN_FLAG=""
    if [ "$MODE" == "frozen" ]; then
        FROZEN_FLAG="--frozen"
    fi

    WEIGHT_TAG=$( [ -z "$CKPT" ] && echo "timm" || basename "${CKPT%.*}" )
    EXP_NAME="${MODEL}_res${RES}_${WEIGHT_TAG}_${MODE}" # Ajout du mode dans le nom
    EXP_DIR="$OUTPUT_BASE/$EXP_NAME"
    mkdir -p "$EXP_DIR"

    # Batch Size dynamique
    BS=16
    if [ "$RES" -eq 518 ]; then BS=4; fi
    if [ "$RES" -eq 384 ]; then BS=8; fi

    echo ">>> Starting: $EXP_NAME (Mode: $MODE | BS: $BS)"

    # --- Step 1: Training ---
    python train.py \
        --model "$MODEL" \
        --img-size "$RES" \
        --checkpoint "$CKPT" \
        --root-dir "$IMG_PARENT_DIR" \
        --metadata "$METADATA_TRAIN" \
        --train-split "train" \
        --val-split "val" \
        --output-dir "$EXP_DIR" \
        --label-col "$LABEL_COL" \
        --workers "$WORKERS" \
        --amp \
        --batch-size "$BS" \
        $FROZEN_FLAG \
        --epochs 50

    # --- Step 2: Evaluation ---
    python test.py \
        --model "$MODEL" \
        --img-size "$RES" \
        --checkpoint "$EXP_DIR/model_best.pth" \
        --root-dir "$IMG_PARENT_DIR" \
        --metadata-train "$METADATA_TRAIN" \
        --metadata-test "$METADATA_TEST" \
        --test-split "test" \
        --label-col "$LABEL_COL" \
        --workers "$WORKERS" \
        --batch-size "$BS" \
        --results-dir "$EXP_DIR"

    # --- Step 3: Visualization ---
    python visualize_umap.py --results-dir "$EXP_DIR" --split test
    python visualize_comparison.py --results-dir "$EXP_DIR"

    
    # --- Step 4: Global CSV Logging (Correction des clés JSON) ---
        GLOBAL_LOG="$RESULT_ROOT/global_results.csv"
        [ ! -f "$GLOBAL_LOG" ] && echo "dataset,model,mode,res,obs_lp_top1,obs_knn_top1" > "$GLOBAL_LOG"
        
        python -c "
import json
try:
    with open('$EXP_DIR/test_metrics.json', 'r') as f:
        d = json.load(f)
        lp = d['linear_probing']['observation']['top1']
        knn = d['knn']['observation']['top1']
        print(f'$DATASET,$MODEL,$MODE,$RES,{lp:.2f},{knn:.2f}')
except Exception: pass
" >> "$GLOBAL_LOG"

        echo "✅ Finished $EXP_NAME"
        echo "----------------------------------------------------------"
    done
done