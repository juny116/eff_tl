""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import json
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric, DatasetDict
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
    parser.add_argument(
        '--save_threshold', 
        default=0, 
        type=int, 
        help='Number of prompt tokens.'
    )

    ## OURS ##
    parser.add_argument(
        '--apply_reverse', 
        default=False, 
        action="store_true",
        help='apply reverse inputs.'
    )
    parser.add_argument(
        '--apply_head', 
        default=False, 
        action="store_true",
        help='apply linear probing.'
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
        writer = SummaryWriter(args.output_dir)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = DatasetDict()
        if args.max_train_samples is not None:
            raw_train_dataset = load_dataset("glue", args.task_name, split=f'train[:{args.max_train_samples}]')
        else:
            raw_train_dataset = load_dataset("glue", args.task_name, split=f'train')
        # Since glue test set is not opened, use 1K train as validation and original validation as test
        
        # for small datasets (RTE, ...)
        if len(raw_train_dataset) < 10000:
            raw_eval_dataset = load_dataset("glue", args.task_name, split=f'validation')
            eval_test_split = raw_eval_dataset.train_test_split(test_size=0.5)
            raw_datasets['train'] = raw_train_dataset
            raw_datasets['validation'] = eval_test_split['train']
            raw_datasets['test'] = eval_test_split['test']
        # for larger datasets
        else:
            train_test_split = raw_train_dataset.train_test_split(test_size=1000)
            raw_datasets['train'] = train_test_split['train']
            raw_datasets['validation'] = train_test_split['test']
            
            # for mnli 
            if args.task_name == "mnli":
                raw_datasets['test'] = load_dataset("glue", args.task_name, split='validation_matched')
            # other tasks
            else:
                raw_datasets['test'] = load_dataset("glue", args.task_name, split=f'validation')
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    if args.local_rank == 0:
        logger.info('TRAIN / VALIDATION / TEST split.')
        for split, dataset in raw_datasets.items():
            logger.info(f'{split} > {len(dataset)}')

    # Labels
    if args.task_name is not None:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
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
        apply_reverse=args.apply_reverse, apply_head=args.apply_head
    )

    # TODO : fix?
    if args.is_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.ds_config):
            model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path)
    else:
        model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path)

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

        input_ids = result['input_ids']

        reversed_input_ids = []
        for input_id in input_ids:
            reversed_input_id = input_id.copy()
            reversed_input_id.reverse()
            reversed_input_ids.append(reversed_input_id)
        result["reversed_input_ids"] = reversed_input_ids

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
    test_dataset = processed_datasets["test"]

    if args.local_rank == 0:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        """ FOR ANALYSIS
        labels2count = {}
        for dataset in train_dataset:
            label = dataset['labels']
            labels2count[label] = labels2count.get(label, 0) + 1
        logger.info(f'TRAIN splits : {labels2count}')

        labels2count = {}
        for dataset in eval_dataset:
            label = dataset['labels']
            labels2count[label] = labels2count.get(label, 0) + 1
        logger.info(f'VALID splits : {labels2count}')

        labels2count = {}
        for dataset in test_dataset:
            label = dataset['labels']
            labels2count[label] = labels2count.get(label, 0) + 1
        logger.info(f'TEST splits : {labels2count}')
        """

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    eval_sampler = DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)       

    # math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric('glue', args.task_name, num_process=args.world_size, process_id=args.local_rank)
    else:
        metric = load_metric("accuracy", num_process=args.world_size, process_id=args.local_rank)

    # Set params to train
    trainable_param_names = []
    if args.apply_lora:
        trainable_param_names.append('lora')
    if args.apply_prefix:
        trainable_param_names.append('prefix')
    if args.apply_adapter:
        trainable_param_names.append('adapter')
    if args.apply_reverse:
        trainable_param_names.append('reverse')
    if args.apply_head:
        trainable_param_names.append('head')

    # if no trainable_param_names -> full fine tune
    if len(trainable_param_names) > 0:
        for name, param in model.named_parameters():
            # train main model? (== fine-tuning)
            if name.startswith('transformer'):
                param.requires_grad = False
                for trainable_param_name in trainable_param_names:
                    if trainable_param_name in name:
                        if args.local_rank == 0:
                            logger.info(f'>> TRAIN {name} {param.shape} -> {param.numel()}')
                        param.requires_grad = True
            else:
                # train PLM encoder?
                if "input_processor.encoder." in name:
                    if args.freeze_encoder:
                        param.requires_grad = False
                    else: 
                        param.requires_grad = True
                        if args.local_rank == 0:
                            logger.info(f'>> TRAINED ENCODER {name} {param.shape} -> {param.numel()}')
                else:
                    param.requires_grad = True
                    if args.local_rank == 0:
                        logger.info(f'>> OTHERS {name} {param.shape} -> {param.numel()}')
                
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad==True],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad==True],
            "weight_decay": 0.0,
        },
    ]

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
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, config_params=args.ds_config)
    # model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config_params=args.ds_config)
    # Train!
    if args.local_rank == 0:
        total_batch_size = args.per_device_batch_size * args.world_size * args.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  World Size = {args.world_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Random Seed = {args.seed}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Number of trainable params = {num_trainable_params}")
        logger.info(f"  Number of total params = {num_total_params}")
        logger.info(f"  % of trainable params = {(100 * num_trainable_params/num_total_params):.3f}")

    # # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=(args.local_rank != 0))
    completed_steps = 0
    best_acc = 0
    ealrt_stop_cnt = 0
    save_flag = False
    for epoch in range(args.num_train_epochs):
        if ealrt_stop_cnt >= args.early_stop:
            break
        model_engine.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss, _ = model_engine(**batch)
            loss = loss / args.gradient_accumulation_steps
            if args.local_rank == 0:
                writer.add_scalar('Train/Loss', loss, model_engine.global_steps)
                writer.add_scalar('Train/LR', model_engine.get_lr()[0], model_engine.global_steps)
            model_engine.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # model step manages optimizer
                model_engine.step()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model_engine.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.cuda() for k, v in batch.items()}
                loss, predictions = model_engine(**batch)
                
                metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )
        eval_metric = metric.compute()
        if args.local_rank == 0:
            writer.add_scalar('Validation/Accuracy', eval_metric['accuracy'], model_engine.global_steps)
            if "f1" in eval_metric.keys():
                writer.add_scalar('Validation/F1', eval_metric['f1'], model_engine.global_steps)
            logger.info(f"Valditaion step {model_engine.global_steps} results {eval_metric}")
            if eval_metric['accuracy'] > best_acc:
                # TODO : save only the models greater than the threshold accuracy
                best_acc = eval_metric['accuracy']
                if best_acc > args.save_threshold:
                    save_flag = True      
                else:
                    save_flag = False      
            else:
                save_flag = False
        
        # path, key, value, current rank, writer rank
        set_value_to_shared_json_file(args.output_dir, 'save_flag', save_flag, args.local_rank, 0)
        save_flag = get_value_from_shared_json_file(args.output_dir, 'save_flag')
        if save_flag:
            model_engine.save_checkpoint(args.output_dir)
            ealrt_stop_cnt = 0
        else:
            ealrt_stop_cnt += 1

    
    # load best dev model 
    # TODO: In ZeRO3 load checkpoint after save checkpoint do not work!!
    if not args.is_zero3:
        try:
            model_engine.load_checkpoint(args.output_dir)
        except:
            logger.info('ERROR LOADING MODEL FOR FINAL TEST. RETRY.')
            model_engine.load_checkpoint(args.output_dir)
        model_engine.eval()
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                batch = {k: v.cuda() for k, v in batch.items()}
                _, predictions = model_engine(**batch)
                metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )
        test_metric = metric.compute()
        if args.local_rank == 0:
            writer.add_scalar('Test/Accuracy', test_metric['accuracy'])
            if "f1" in test_metric.keys():
                writer.add_scalar('Test/F1', test_metric['f1'], model_engine.global_steps)
            logger.info(f"TEST results {test_metric}")

if __name__ == "__main__":
    main()