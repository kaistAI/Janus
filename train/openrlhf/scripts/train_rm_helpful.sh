set -x 

mkdir -p ./ckpt/7b_mistral_66k_rm_helpful
mkdir -p ./log

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG=WARN

WANDB_API_KEY=<YOUR_WANDB_API_KEY>
WANDB_ENTITY=<YOUR_WANDB_ENTITY>
WANDB_PROJECT=<YOUR_WANDB_PROJECT>
WANDB_RUN_NAME=<YOUR_WANDB_RUN_NAME>

PRETRAIN_MODEL_PATH=./ckpt/7b_mistral_66k_rm  # or your own <REWARD_MODEL_PATH>
MODEL_OUTPUT_PATH=./ckpt/7b_mistral_66k_rm_helpful


read -r -d '' training_commands <<EOF
../../train_rm.py \
     --save_path $MODEL_OUTPUT_PATH \
     --save_steps 500 \
     --logging_steps 1 \
     --eval_steps 50 \
     --max_ckpt_num 2 \
     --train_batch_size 128 \
     --micro_train_batch_size 8 \
     --pretrain $PRETRAIN_MODEL_PATH \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --max_samples 66000 \
     --dataset Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward,openai/webgpt_comparisons \
     --dataset_probs 0.72,0.14,0.14 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB_API_KEY \
     --wandb_org $WANDB_ENTITY \
     --wandb_project $WANDB_PROJECT \
     --wandb_run_name $WANDB_RUN_NAME
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # micro_train_batch_size 16 OOMs on 4xA100 80GB gpu


if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port 29501 $training_commands
fi
