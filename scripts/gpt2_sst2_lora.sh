export CUDA_VISIBLE_DEVICES=0,1,2
export num_gpus=3
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./sst2_gpt2_medium"
#export output_dir="./sst2_gpt"
export TORCH_DISTRIBUTED_DEBUG=INFO

learning_rates="1e-5"

for learning_rate in $learning_rates; do
    python -m torch.distributed.launch --nproc_per_node=$num_gpus \
        run_glue.py \
        --model_name_or_path gpt2-medium \
        --task_name sst2 \
        --do_train \
        --do_eval \
        --do_predict \
        --max_seq_length 128 \
        --per_device_train_batch_size 8 \
        --learning_rate $learning_rate \
        --num_train_epochs 20 \
        --output_dir $output_dir/lora/$learning_rate \
        --overwrite_output_dir \
        --logging_steps 10 \
        --logging_dir $output_dir/lora/$learning_rate \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --warmup_ratio 0.06 \
        --seed 0 \
        --apply_lora \
        --lora_r 8 \
        --lora_alpha 16 \
        --save_total_limit 1
done