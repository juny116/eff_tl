
deepspeed main.py \
    --task_name mnli \
    --model_name_or_path gpt2 \
    --ds_config ds_configs/fp16.json \
    --output_dir /home/heyjoonkim/data/heyjoonkim/prompt_test/FT \
    --seed 1234 \
    --max_train_samples 40000