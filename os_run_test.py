import os

os.system("deepspeed main.py --task_name sst2 --model_name_or_path gpt2 --ds_config ds_configs/fp16.json --output_dir /home/juny116/data/gpt2_test --seed 1234 --apply_lora")