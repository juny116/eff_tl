export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./sst2_gpt"
export TORCH_DISTRIBUTED_DEBUG=INFO
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
run_glue.py \
--model_name_or_path gpt2 \
--task_name sst2 \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 8 \
--learning_rate 5e-3 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--seed 0 \
--weight_decay 0.1
