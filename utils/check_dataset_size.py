

from datasets import load_dataset

# benchmark = "glue"
benchmark = "super_glue"

splits = ["train", "validation", "test"]

# tasks=["mrpc", "stsb", "qqp", "qnli", "rte", "wnli", "sst2"]
tasks=["axg","boolq","cb","copa","axb","multirc","record","rte","wic","wsc","wsc.fixed"]

for task in tasks:
    for split in splits:
        try:
            datasets = load_dataset(benchmark, task, split=split)
            print(f'{task} | {split} | {len(datasets)}')
        except:
            print("ERROR. SKIP.")
    print('=' * 100)


