export CUDA_VISIBLE_DEVICES=0,1,2

main_model="gpt2-xl"

task="mnli"
#learning_rates="1e-4 5e-5 1e-5"
learning_rates="1e-6"

seed="1234"
train_epochs="40"

path="/home/heyjoonkim/data/heyjoonkim/prompt_test"

# Trained IDPG
for learning_rate in $learning_rates; do
    deepspeed main.py \
        --task_name $task \
        --model_name_or_path $main_model \
        --ds_config ds_configs/zero2_config.json \
        --output_dir $path/$task/IDPG-trained/$learning_rate/ \
        --seed $seed \
        --num_train_epochs $train_epochs \
        --max_train_samples 40000 \
        --apply_encoder \
        --apply_input \
        --encoder_model_name_or_path gpt2 \
        --prompt_length 20
done