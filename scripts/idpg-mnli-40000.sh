
learning_rates="1e-4 5e-5 1e-5"

learning_rates="1e-4"

# original IDPG
for learning_rate in $learning_rates; do
    deepspeed main.py \
        --task_name mnli \
        --model_name_or_path gpt2-xl \
        --ds_config ds_configs/zero2_config.json \
        --output_dir /home/heyjoonkim/data/heyjoonkim/prompt_test/IDPG-freezed/$learning_rate/ \
        --seed 1234 \
        --num_train_epochs 40 \
        --max_train_samples 40000 \
        --apply_encoder \
        --apply_input \
        --encoder_model_name_or_path gpt2 \
        --freeze_encoder \
        --prompt_length 20
done

# Trained IDPG
for learning_rate in $learning_rates; do
    deepspeed main.py \
        --task_name mnli \
        --model_name_or_path gpt2-xl \
        --ds_config ds_configs/zero2_config.json \
        --output_dir /home/heyjoonkim/data/heyjoonkim/prompt_test/IDPG-trained/$learning_rate/ \
        --seed 1234 \
        --max_train_samples 40000 \
        --apply_encoder \
        --apply_input \
        --encoder_model_name_or_path gpt2 \
        --prompt_length 20
done