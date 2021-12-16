main_model="gpt2-xl"

task="mnli"
learning_rates="1e-4 5e-5 1e-5"
seed="1234"
train_epochs="60"

# NOT NOW
# Trained IDPG
# for learning_rate in $learning_rates; do
#     deepspeed main.py \
#         --task_name $task \
#         --model_name_or_path $main_model \
#         --ds_config ds_configs_samples/zero2_config.json \
#         --output_dir $output_dir/$task/IDPG-trained/$learning_rate/ \
#         --seed $seed \
#         --lr $learning_rate \
#         --num_train_epochs $train_epochs \
#         --max_train_samples 40000 \
#         --apply_encoder \
#         --apply_input \
#         --encoder_model_name_or_path gpt2 \
#         --prompt_length 20
# done


learning_rates="5e-4 1e-3"
# original IDPG w/ only prompt inputs
# for learning_rate in $learning_rates; do
#     deepspeed main.py \
#         --task_name $task \
#         --model_name_or_path $main_model \
#         --ds_config ds_configs_samples/zero2_config.json \
#         --output_dir $output_dir/$task/IDPG-prompt-only-freezed/$learning_rate/ \
#         --seed $seed \
#         --lr $learning_rate \
#         --num_train_epochs $train_epochs \
#         --max_train_samples 40000 \
#         --apply_encoder \
#         --encoder_model_name_or_path gpt2 \
#         --freeze_encoder \
#         --prompt_length 20
# done