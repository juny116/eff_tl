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
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    set_seed,
)
import torch
import deepspeed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model_wrapper.GPT2Wrapper import GPT2Wrapper
from utils import save_config, set_value_to_shared_json_file, get_value_from_shared_json_file
from dataset_utils import sst5_generate_dataset_dict

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
    "sst5": ("sentence", None),
}

task_to_path = {
    "sst5" : {
        "train" : "/home/heyjoonkim/data/datasets/sst5/train.csv",
        "validation" : "/home/heyjoonkim/data/datasets/sst5/test.csv",
        "dataset_processor" : sst5_generate_dataset_dict,
    },
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
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
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
        "--manual_prompt",
        type=str,
        help="Manual prompt to append at the end.",
        required=False,
    )
    # FOR GENERATION
    parser.add_argument(
        '--generation_length', 
        default=10, 
        type=int, 
        help='Max length for generation.'
    )
    parser.add_argument(
        '--num_beam', 
        default=5, 
        type=int, 
        help='Beam size.'
    )
    parser.add_argument(
        '--num_return_sequences', 
        default=5, 
        type=int, 
        help='Number of sequences generated.'
    )
    parser.add_argument(
        '--no_repeat_ngram_size', 
        default=2, 
        type=int, 
        help='no_repeat_ngram_size.'
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50, 
        help="For top-k sampling."
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.95, 
        help="For top-p sampling"
    )
    parser.add_argument(
        "--generated_file_name",
        type=str,
        default="generation_results"
    )
    args = parser.parse_args()
    
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.task_name in task_to_path:
            args.train_file = task_to_path[args.task_name]['train']
            args.validation_file = task_to_path[args.task_name]['validation']

        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

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
    if args.task_name is not None and args.task_name not in task_to_path:
    # if args.train_file is None and args.validation_file is None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = DatasetDict()
        if args.max_train_samples is not None:
            raw_train_dataset = load_dataset("glue", args.task_name, split=f'train[:{args.max_train_samples}]')
        else:
            raw_train_dataset = load_dataset("glue", args.task_name, split=f'train')

        # Since glue test set is not opened, we only use train and validation set
        raw_datasets['train'] = raw_train_dataset
        
        # for mnli 
        if args.task_name == "mnli":
            raw_datasets['validation'] = load_dataset("glue", args.task_name, split='validation_matched')
        # other tasks
        else:
            raw_datasets['validation'] = load_dataset("glue", args.task_name, split=f'validation')
    else:

        dataset_processor = task_to_path[args.task_name]["dataset_processor"]

        # Loading the dataset from local csv or json file.
        raw_datasets = DatasetDict()
        if args.train_file is not None:
            train_dict = dataset_processor(args.train_file)
            train_dataset = Dataset.from_dict(train_dict)
            raw_datasets['train'] = train_dataset
        if args.validation_file is not None:
            validation_dict = dataset_processor(args.validation_file)
            validation_dataset = Dataset.from_dict(validation_dict)
            raw_datasets['validation'] = validation_dataset

    if args.local_rank == 0:
        logger.info('TRAIN / VALIDATION split.')
        for split, dataset in raw_datasets.items():
            logger.info(f'{split} > {len(dataset)}')

    # Labels
    if args.task_name is not None and args.task_name not in task_to_path:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    # For gpt-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # TODO: only inject pad_token_id in case of GPT
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, 
        finetuning_task=args.task_name, pad_token_id=tokenizer.unk_token_id
    )

    # TODO : we use GPT2LMHeadModel for generation
    if args.is_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.ds_config):
            # model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path)
            model= AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        # model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path)
        model= AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

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
        if args.local_rank == 0:
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

        if args.manual_prompt:
            logger.info(f'Generate input with manual prompt : {args.manual_prompt}')
            manual_prompt = tokenizer(args.manual_prompt)

            result['input_ids'] = [ids + manual_prompt['input_ids'] for ids in result['input_ids']]
            result['attention_mask'] =  [mask + manual_prompt['attention_mask'] for mask in result['attention_mask']]

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    if args.local_rank != 0:
        torch.distributed.barrier()
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)
    eval_sampler = DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)
    

    if args.local_rank == 0:
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in model.parameters())
        transformer_params = sum(p.numel() for n,p in model.named_parameters() if n.startswith('transformer'))
        logger.info(f'trainable params {num_trainable_params} / total params {num_total_params} = ratio {100 * num_trainable_params/num_total_params} ')
        
        ## Write parameter info ##
        parameter_summary_file = os.path.join(args.output_dir, "parameter_summary.txt")
        with open(parameter_summary_file, "w") as file_writer:
            file_writer.write("Overall Parameter Summary\n")
            file_writer.write(f"Trained     parameters\t{num_trainable_params}\n")
            file_writer.write(f"Transformer parameters\t{transformer_params}\n")
            file_writer.write(f"Total       parameters\t{num_total_params}\n")
            file_writer.write(f"Trainable   ratio\t\t{100 * num_trainable_params / num_total_params} \n")
            file_writer.write("=" * 50 + '\n')
            file_writer.write("Trained parameters detail\n")

            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    file_writer.write(f"{name} > {param.shape} \n")
    

    model_engine, _, _, _ = deepspeed.initialize(model=model, config_params=args.ds_config)
    # model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config_params=args.ds_config)
    
    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
        logger.info(f"  World Size = {args.world_size}")
        logger.info(f"  Random Seed = {args.seed}")
        
    
    # No parameter updates, only generation.
    model_engine.eval()

    # ignore generating comma(,) and new_line(\n)
    ignored_sequences = [',', ' ,', ' \n', '\n', ' \t', '\t']
    # bad_words_ids = tokenizer(ignored_sequences, add_prefix_space=True).input_ids
    # bad_words_ids = [ tokenizer.encode(ignored_sequence, add_prefix_space=True) for ignored_sequence in ignored_sequences]
    bad_words_ids = [ tokenizer.encode(ignored_sequence, add_prefix_space=True) for ignored_sequence in ignored_sequences]
    print(bad_words_ids)

    # progress_bar = tqdm(range(len(train_dataloader)), disable=(args.local_rank != 0))
    progress_bar = tqdm(range(len(train_dataloader)), disable=(args.local_rank != 0))
    total_index = 0
    # write_path = os.path.join(args.output_dir, args.model_name_or_path)
    generation_writer = os.path.join(args.output_dir, "train_samples_generation_results.tsv")
    with open(generation_writer, "w") as file_writer:
        tsv_writer = csv.writer(file_writer, delimiter='\t')
        tsv_writer.writerow(['sample index', 'label', 'input', 'generation1', 'generation2', 'generation3'])
        # file_writer.write('sample index\tlabel\tinput\tgeneration1\tgeneration2\tgeneration3')
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = batch['labels']
            input_ids = batch['input_ids']
            batch_length, input_length = input_ids.shape
            original_input = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

            # max_length includes the input sentence length
            # for beam search
            # generated_output = model_engine.module.generate(
            #     **batch, 
            #     max_length=input_length+args.generation_length,
            #     num_beams=args.num_beam,
            #     no_repeat_ngram_size=args.no_repeat_ngram_size,
            #     num_return_sequences=args.num_return_sequences,
            #     early_stopping=True,
            #     bad_words_ids=bad_words_ids
            # )

            generated_output_ids = model_engine.module.generate(
                **batch, 
                do_sample=True,
                max_length=input_length+args.generation_length,
                top_k=args.top_k,
                top_p=args.top_p,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                num_return_sequences=args.num_return_sequences,
                early_stopping=True,
                bad_words_ids=bad_words_ids
            )

            # generated_output = generated_output.reshape(batch_length, args.num_return_sequences, -1)

            # shape : (batch * num_return_sequences, generated_length)
            generated_output = tokenizer.batch_decode(generated_output_ids, skip_special_tokens=True)

            for batch_index in range(batch_length):
                data = [total_index, labels[batch_index].item(), original_input[batch_index]]
                
                # file_writer.write(f'{input_ids[batch_index].tolist()}\n')
                input_length = len(original_input[batch_index])
                for generation_index in range(args.num_return_sequences):
                    full_generated_output = generated_output[batch_index * args.num_return_sequences + generation_index]
                    full_generated_output = full_generated_output[input_length:]
                    for ignored_sequence in ignored_sequences:
                        full_generated_output = full_generated_output.replace(ignored_sequence, ' ')
                    # file_writer.write(f'\t{full_generated_output}')
                    data.append(full_generated_output)
                    # file_writer.write(f'{generated_output_ids[batch_index * args.num_return_sequences + generation_index]}\n')
                
                total_index += 1
                tsv_writer.writerow(data)
            progress_bar.update(1)

    # logger.info('Done.')
    # exit()

    progress_bar = tqdm(range(len(eval_dataloader)), disable=(args.local_rank != 0))
    total_index = 0
    # write_path = os.path.join(args.output_dir, args.model_name_or_path)
    generation_writer = os.path.join(args.output_dir, "test_samples_generation_results.tsv")
    with open(generation_writer, "w") as file_writer:
        tsv_writer = csv.writer(file_writer, delimiter='\t')
        tsv_writer.writerow(['sample index', 'label', 'input', 'generation1', 'generation2', 'generation3'])
        # file_writer.write('sample index\tlabel\tinput\tgeneration1\tgeneration2\tgeneration3')
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = batch['labels']
            input_ids = batch['input_ids']
            batch_length, input_length = input_ids.shape
            original_input = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

            # max_length includes the input sentence length
            # for beam search
            # generated_output = model_engine.module.generate(
            #     **batch, 
            #     max_length=input_length+args.generation_length,
            #     num_beams=args.num_beam,
            #     no_repeat_ngram_size=args.no_repeat_ngram_size,
            #     num_return_sequences=args.num_return_sequences,
            #     early_stopping=True,
            #     bad_words_ids=bad_words_ids
            # )

            generated_output_ids = model_engine.module.generate(
                **batch, 
                do_sample=True,
                max_length=input_length+args.generation_length,
                top_k=args.top_k,
                top_p=args.top_p,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                num_return_sequences=args.num_return_sequences,
                early_stopping=True,
                bad_words_ids=bad_words_ids
            )

            # generated_output = generated_output.reshape(batch_length, args.num_return_sequences, -1)

            # shape : (batch * num_return_sequences, generated_length)
            generated_output = tokenizer.batch_decode(generated_output_ids, skip_special_tokens=True)

            for batch_index in range(batch_length):
                data = [total_index, labels[batch_index].item(), original_input[batch_index]]
                
                # file_writer.write(f'{input_ids[batch_index].tolist()}\n')
                input_length = len(original_input[batch_index])
                for generation_index in range(args.num_return_sequences):
                    full_generated_output = generated_output[batch_index * args.num_return_sequences + generation_index]
                    full_generated_output = full_generated_output[input_length:]
                    for ignored_sequence in ignored_sequences:
                        full_generated_output = full_generated_output.replace(ignored_sequence, ' ')
                    # file_writer.write(f'\t{full_generated_output}')
                    data.append(full_generated_output)
                    # file_writer.write(f'{generated_output_ids[batch_index * args.num_return_sequences + generation_index]}\n')
                
                total_index += 1
                tsv_writer.writerow(data)
            progress_bar.update(1)
        

if __name__ == "__main__":
    main()