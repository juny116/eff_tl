""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import csv
import json
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric, DatasetDict, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    get_scheduler,
    set_seed,
)
import torch
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint, load_state_dict_from_zero_checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model_wrapper.GPT2Wrapper import GPT2Wrapper
from utils import save_config, set_value_to_shared_json_file, get_value_from_shared_json_file

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_train_samples",
        default=None,
        help="Maximum train samples to use at train time, slice from raw train dataset for fast experiment purpose",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=20, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--early_stop", 
        type=int, 
        default=5, 
        help="Number of epoch for early stopping."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=1000, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        '--overwrite_output_dir', 
        default=False, 
        action="store_true",
        help='Overwrite output directory.'
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        '--ds_config', 
        default='ds_config.json', 
        type=str, 
        help='deepspeed config'
    )
    parser.add_argument(
        '--local_rank', 
        default=0, 
        type=int, 
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--apply_lora', 
        default=False, 
        action="store_true",
        help='apply LoRA params'
    )
    parser.add_argument(
        '--lora_alpha', 
        default=16, 
        type=int, 
        help='LoRA alpha'
    )
    parser.add_argument(
        '--lora_r', 
        default=8, 
        type=int, 
        help='LoRA r'
    )
    parser.add_argument(
        '--apply_prefix', 
        default=False, 
        action="store_true",
        help='apply prefix tuning params'
    )
    parser.add_argument(
        '--num_prefix', 
        default=10, 
        type=int, 
        help='number of prefix to append per layer'
    )
    parser.add_argument(
        '--mid_dim', 
        default=16, 
        type=int, 
        help='reparameterization dim'
    )
    parser.add_argument(
        '--apply_adapter', 
        default=False, 
        action="store_true",
        help='apply adapter tuning params'
    )

    ## OURS ##
    parser.add_argument(
        '--apply_encoder', 
        default=False, 
        action="store_true",
        help='Apply input dependent encoder.'
    )
    parser.add_argument(
        '--apply_input', 
        default=False, 
        action="store_true",
        help='Apply input for prompt generating.'
    )
    parser.add_argument(
        '--encoder_model_name_or_path', 
        default='gpt2', 
        type=str, 
        help='PLM for encoder.'
    )
    parser.add_argument(
        '--freeze_encoder', 
        default=False, 
        action="store_true",
        help='Freeze PLM for the encoder.'
    )
    parser.add_argument(
        '--apply_prompt', 
        default=False, 
        action="store_true",
        help='apply prompt tuning'
    )
    parser.add_argument(
        '--prompt_length', 
        default=None, 
        type=int, 
        help='Number of prompt tokens.'
    )
    parser.add_argument(
        '--reparameterize', 
        default=False, 
        action="store_true",
        help='Reparameterize prompt.'
    )
    parser.add_argument(
        '--apply_ptuning', 
        default=False, 
        action="store_true",
        help='Apply P-tuning.'
    )
    parser.add_argument(
        '--save_threshold', 
        default=0, 
        type=int, 
        help='Number of prompt tokens.'
    )
    # for loading trained model
    parser.add_argument(
        "--pretrained_dir", 
        type=str, 
        default=None, 
        help="Where the pretrained model is."
    )
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default=None, 
        help="Where the dataset file is."
    )

    args = parser.parse_args()
    
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "tsv"], "`validation_file` should be a csv/json/tsv file."

    # post init get batch and zero option from ds config
    with open(args.ds_config, "r", encoding="utf-8") as ds_f:
        ds_config = json.load(ds_f)
    args.per_device_batch_size = ds_config['train_micro_batch_size_per_gpu']
    args.gradient_accumulation_steps = ds_config['gradient_accumulation_steps']
    if ds_config.get("zero_optimization"):
        args.is_zero3 = ds_config["zero_optimization"]["stage"] == 3
    else:
        args.is_zero3 = False

    return args


def main():
    args = parse_args()
    dschf = HfDeepSpeedConfig(args.ds_config)
    deepspeed.init_distributed()
    args.world_size = torch.distributed.get_world_size()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if args.local_rank == 0 else logging.ERROR)

    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.overwrite_output_dir:
                logger.info(f'Output directory {args.output_dir} exits. Exit program. (overwrite_output_dir=False)')
                exit()
            
    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if args.local_rank == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation & SummaryWriter
    if args.local_rank == 0:
        # if args.output_dir is not None:
        #     os.makedirs(args.output_dir, exist_ok=True)
        save_config(args)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        raise NotImplementedError('This file is for validation of customized dataset.')
    else:
        # Loading the dataset from local csv or json file.
        raw_datasets = DatasetDict()
        if args.train_file is not None:
            pass
        if args.validation_file is not None:
            input_list = []
            label_list = []
            index_list = []
            with open(args.validation_file) as f:
                validation_lines = csv.reader(f, delimiter='\t')
                # Remove header
                next(validation_lines, None)

                for validation_line in validation_lines:
                    sample_index = int(validation_line[0])
                    label = int(validation_line[1])
                    input_sentence = validation_line[2]
                    prompt = validation_line[3]

                    input_sentence = '.'.join([input_sentence, prompt])

                    label_list.append(label)
                    input_list.append(input_sentence)
                    index_list.append(sample_index)
            validation_dict = {
                'sample_index' : index_list,
                'sentence' : input_list,
                'label' : label_list,
            }
            validation_dataset = Dataset.from_dict(validation_dict)
            raw_datasets['validation'] = validation_dataset

    logger.info('TEST split.')
    for split, dataset in raw_datasets.items():
        logger.info(f'{split} > {len(dataset)}')


    # Labels
    if args.task_name is not None:
        label_list = raw_datasets["test"].features["label"].names
        num_labels = len(label_list)
    else:
        label_list = set(raw_datasets["validation"]["label"])
        logger.info(label_list)
        num_labels = len(label_list)
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # For gpt-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # TODO: only inject pad_token_id in case of GPT
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, 
        finetuning_task=args.task_name, pad_token_id=tokenizer.unk_token_id,
        apply_lora=args.apply_lora, lora_alpha=args.lora_alpha, lora_r=args.lora_r,
        apply_prefix=args.apply_prefix, num_prefix=args.num_prefix, mid_dim=args.mid_dim,
        apply_encoder=args.apply_encoder, apply_input=args.apply_input, encoder_model_name_or_path=args.encoder_model_name_or_path,
        freeze_encoder=args.freeze_encoder, prompt_length=args.prompt_length,
        reparameterize=args.reparameterize, apply_ptuning=args.apply_ptuning,
    )

    # TODO : fix?
    if args.is_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.ds_config):
            model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path)
    else:
        model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path)

    

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        sentence1_key = "sentence"
        sentence2_key = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None:
        logger.info('Auto label2id, id2label created')
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

        if "sample_index" in examples:
                result["sample_index"] = examples["sample_index"]

        return result

    if args.local_rank != 0:
        torch.distributed.barrier()
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Running tokenizer on dataset",
    )
    if args.local_rank == 0:
        torch.distributed.barrier()

    assert "validation" in processed_datasets.keys(), f'validation set not in dataset, got :  {processed_datasets.keys()}'
    test_dataset = processed_datasets["validation"]

    # For analysis
    labels2count = {}
    for dataset in test_dataset:
        label = dataset['labels']
        labels2count[label] = labels2count.get(label, 0) + 1
    logger.info(f'TEST splits : {labels2count}')


    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)       


    # Get the metric function
    if args.task_name is not None:
        metric = load_metric('glue', args.task_name, num_process=args.world_size, process_id=args.local_rank)
    else:
        metric = load_metric("accuracy", num_process=args.world_size, process_id=args.local_rank)


    model_engine, _, _, _ = deepspeed.initialize(model=model, config_params=args.ds_config)
    # model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config_params=args.ds_config)
    

    logger.info(f'Load model from checkpoint : {args.pretrained_dir}')
    model_engine.load_checkpoint(load_dir=args.pretrained_dir, load_module_only=True, load_module_strict=False)
    logger.info('Done loading model.')
    
    # Train!
    if args.local_rank == 0:
        total_batch_size = args.per_device_batch_size * args.world_size * args.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(test_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
        logger.info(f"  World Size = {args.world_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Random Seed = {args.seed}")

    index = 0
    wrong_answer_file_path = os.path.join(args.output_dir, "wrong_prediction.txt")
    with open(wrong_answer_file_path, "w") as file_writer:
        file_writer.write('index | input | label | prediction\n')
        model_engine.eval()
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                batch = {k: v.cuda() for k, v in batch.items()}
                loss, predictions = model_engine(**batch)

                batch_size = predictions.size(dim=0)
                
                metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )
                
                for batch_index in range(batch_size):
                    label = batch["labels"][batch_index]
                    prediction = predictions[batch_index]
                    if label != prediction:
                        original_input = tokenizer.decode(batch['input_ids'][batch_index], skip_special_tokens=True)
                        file_writer.write(f'{batch["sample_index"][batch_index].item()} | {original_input} | {label} | {prediction} \n')
                    index += 1



        eval_metric = metric.compute()
        logger.info(f"Test result : {eval_metric}")

        

if __name__ == "__main__":
    main()