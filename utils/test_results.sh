

task="mnli"
main_model="gpt2"
main_path="/home/heyjoonkim/data/fine_tuining/"

seed="1234"
train_epochs="10"

learning_rate="5e-5"

# linear probing
deepspeed test_results.py \
    --task_name $task \
    --model_name_or_path $main_model \
    --ds_config ds_configs_samples/zero2_config.json \
    --output_dir $main_path/$task/$main_model/$learning_rate/ \
    --seed $seed \
    --lr $learning_rate \
    --num_train_epochs $train_epochs \
    --overwrite_output_dir

