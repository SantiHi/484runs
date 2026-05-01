EDGE_SPARSITIES=(0.94 0.945 0.95 0.955 0.96 0.965 0.97 0.975 0.98 0.985 0.99 0.995 1.0 1.01 1.02 1.05 1.1)

# --- Compression settings ---
COMPRESSION_DIM=64          # Latent dimension k (0 = disabled, same as original)
COMPRESSION_INIT="gaussian" # "gaussian", "orthogonal", or path to pretrained weights
FREEZE_PROJECTIONS="true"   # "true" to freeze P and U during mask optimization

for i in "${!EDGE_SPARSITIES[@]}"; do

EDGE_SPARSITY=${EDGE_SPARSITIES[i]}
NODE_SPARSITY=0.72
ELR=0.8
LLR=0.8
RELR=0.8
RLLR=0.8
TOTAL=3000
WARMUP=2500

EXTRA="--disable_node_loss"
TAG="compressed_k${COMPRESSION_DIM}-wo_node_loss"

# Uncomment this if you want to run with node loss
# EXTRA=""
# TAG="compressed_k${COMPRESSION_DIM}-w_node_loss"

train_split="train" # "train_400", "train_100k"
N_TRAIN=1000000 # Set to a large value so all of the (200 / 400 / 100000) examples are used
N_VAL=200 # The val split size

# Build freeze flag
FREEZE_FLAG=""
if [ "$FREEZE_PROJECTIONS" = "true" ]; then
    FREEZE_FLAG="--freeze_projections"
fi

python src/prune/fpt2_ioi_compressed.py \
    --report_to none \
    --do_train \
    --do_eval \
    --dataset_path ./data/datasets/ioi/ \
    --train_split $train_split \
    --initialize_from gpt2 \
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
    --start_edge_sparsity 0.00 \
    --target_edge_sparsity $EDGE_SPARSITY \
    --start_layer_sparsity 0.00 \
    --target_layer_sparsity $NODE_SPARSITY \
    --num_sparsity_warmup_steps $WARMUP \
    --max_train_samples $N_TRAIN \
    --max_eval_samples $N_VAL \
    --output_dir ./data/runs/ioi-${TAG}-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/ \
    --remove_unused_columns false \
    --dataloader_num_workers 0 \
    --warmup_type linear \
    --with_embedding_nodes \
    --compression_dim $COMPRESSION_DIM \
    --compression_init $COMPRESSION_INIT \
    $FREEZE_FLAG \
    $EXTRA

done
