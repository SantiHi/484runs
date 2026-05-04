#!/bin/bash
# run_scripts/ioi_warmstart.sh
#
# EAP-IG-warmstarted Edge Pruning on IOI.
# Identical to ioi_sweep.sh except:
#   1. --initialize_from points at our warmstart checkpoint (not raw gpt2)
#   2. --start_edge_sparsity is raised to match the warmstart's sparsity
#      (no point starting from 0% if our warmstart already begins at 90%)
#   3. output_dir name is prefixed with "warmstart-" so it doesn't
#      collide with the baseline run's directory
#
# Prereq: warmstart_ckpt/ exists at the path below. Build it from the
# EAP-IG notebook with the bridge module (eap_to_ep_warmstart.py).

EDGE_SPARSITIES=(0.98)

for i in "${!EDGE_SPARSITIES[@]}"; do

EDGE_SPARSITY=${EDGE_SPARSITIES[i]}
NODE_SPARSITY=0.72
ELR=0.8
LLR=0.8
RELR=0.8
RLLR=0.8
TOTAL=1500
WARMUP=1200

EXTRA="--disable_node_loss"
TAG="wo_node_loss"

train_split="train"
N_TRAIN=1000000
N_VAL=200

# Path to the warm-started checkpoint built by the bridge module.
# Adjust this if you saved it somewhere else.
WARMSTART_CKPT="/content/drive/MyDrive/EAP-and-Edge/warmstart/warmstart_ckpt"

python src/prune/fpt2_ioi.py \
    --report_to none \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/ioi/ \
    --train_split $train_split \
    --initialize_from $WARMSTART_CKPT \
    --max_seq_length 64 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 16 \
    --edge_learning_rate $ELR \
    --layer_learning_rate $LLR \
    --reg_edge_learning_rate $RELR \
    --reg_layer_learning_rate $RLLR \
    --max_steps $TOTAL \
    --warmup_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --save_steps 64 \
    --logging_steps 8 \
    --save_total_limit 1 \
    --start_edge_sparsity 0.90 \
    --target_edge_sparsity $EDGE_SPARSITY \
    --start_layer_sparsity 0.00 \
    --target_layer_sparsity $NODE_SPARSITY \
    --num_sparsity_warmup_steps $WARMUP \
    --max_train_samples $N_TRAIN \
    --max_eval_samples $N_VAL \
    --output_dir ./data/runs/ioi-warmstart-${TAG}-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    --with_embedding_nodes \
    $EXTRA

done
