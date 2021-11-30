export CUDA_VISIBLE_DEVICES=0,1,2,3
export num_gpus=4
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./sst2_gpt2_xl"
#export output_dir="./sst2_gpt"
# export TORCH_DISTRIBUTED_DEBUG=INFO

learning_rates="1e-5"

for learning_rate in $learning_rates; do
    python -m torch.distributed.launch --nproc_per_node=$num_gpus \
        main.py \
        --model_name_or_path gpt2-xl \
        --task_name mnli \
        --do_train \
        --do_eval \
        --do_predict \
        --max_seq_length 128 \
        --per_device_train_batch_size 8 \
        --learning_rate $learning_rate \
        --num_train_epochs 2 \
        --output_dir $output_dir/fine-tuning/$learning_rate \
        --overwrite_output_dir \
        --logging_steps 10 \
        --logging_dir $output_dir/fine-tuning/$learning_rate \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --warmup_ratio 0.06 \
        --seed 0 \
        --weight_decay 0.1 \
        --save_total_limit 5 \
        --max_train_samples 40000 
done