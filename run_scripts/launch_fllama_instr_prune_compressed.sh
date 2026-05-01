#!/bin/bash -l
#SBATCH --job-name=instr_prune-fllama-compressed
#SBATCH --nodes=4
#SBATCH --output=./joblog/%x-%A_%a.out
#SBATCH --error=./joblog/%x-%A_%a.err
#SBATCH --gres=gpu:8
#SBATCH --mem=700G
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=16

LOG_DIR=joblog

num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
if [ -z "$SLURM_GPUS_PER_NODE" ]; then
    export SLURM_GPUS_PER_NODE=8
fi
echo $SLURM_GPUS_PER_NODE

export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))
export NUM_NODES=$num_nodes

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

ELR=0.8
LLR=$ELR
RELR=0.4
RLLR=$RELR
EDGE_SPARSITY=1.2
NODE_SPARSITY=0.7
TOTAL=6000
WARMUP=5500
SEED=42

# --- Compression settings ---
COMPRESSION_DIM=256         # Latent dimension k (CodeLlama hidden_size is large, so use larger k)
COMPRESSION_INIT="gaussian"

OUTPUT_DIR=./data/runs/fllama-instr-compressed_k${COMPRESSION_DIM}-s${SEED}-elr${ELR}-llr${LLR}-relr${RELR}-rllr${RLLR}-es${EDGE_SPARSITY}-ns${NODE_SPARSITY}-t${TOTAL}/

mkdir -p $OUTPUT_DIR

srun bash run_scripts/wrapper_launch_fllama_prune.sh \
src/prune/fllama_boolean_expressions_ip_compressed.py \
--report_to none \
--do_train \
--dataset_path ./data/datasets/merged/boolean_expressions_inhouse_big/ \
--initialize_from meta-llama/CodeLlama-13b-Instruct-hf \
--ref_initialize_from meta-llama/CodeLlama-13b-Instruct-hf \
--max_seq_length 64 \
--per_device_train_batch_size 1 \
--edge_learning_rate $ELR \
--node_learning_rate $LLR \
--reg_edge_learning_rate $RELR \
--reg_node_learning_rate $RLLR \
--max_steps $TOTAL \
--warmup_steps 200 \
--save_steps 512 \
--logging_steps 8 \
--save_total_limit 1 \
--start_edge_sparsity 0.00 \
--target_edge_sparsity $EDGE_SPARSITY \
--start_node_sparsity 0.00 \
--target_node_sparsity $NODE_SPARSITY \
--num_sparsity_warmup_steps $WARMUP \
--max_train_samples 8000 \
--output_dir ${OUTPUT_DIR} \
--remove_unused_columns false \
--dataloader_num_workers 0 \
--warmup_type linear \
--bf16 \
--gradient_checkpointing \
--seed $SEED \
--disable_node_loss \
--compression_dim $COMPRESSION_DIM \
--compression_init $COMPRESSION_INIT \
--freeze_projections
