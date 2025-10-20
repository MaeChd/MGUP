# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import wandb
import datasets
import evaluate
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForSequenceClassification
)
from datasets import load_dataset
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from MGUP import MGUP_AdamW as MGAdamW
from MGUP import MGUP_Lion as MGLion
from MGUP import MGUP_Muon as MGMuon
from torch_optimizer import Lion,AdamW
from c_lion import Lion as CLion
from c_muon import Muon as CMuon
from c_adamw import AdamW as CAdamW
from muon import Muon as Muon

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.38.0.dev0")

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

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

# Data from https://huggingface.co/datasets/nyu-mll/glue as custom datasets
task_to_data_dir = {
    "cola": ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
    "mnli": ("train-00000-of-00001.parquet", "validation_matched-00000-of-00001.parquet"),
    "mrpc": ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
    "qnli": ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
    "qqp": ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
    "rte": ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
    "sst2": ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
    "stsb": ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
    "wnli": ("train-00000-of-00001.parquet", "validation-00000-of-00001.parquet"),
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--load_pretrained_model", type=str, default=None)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # circle vector 
    parser.add_argument("--num_vec", type=int, default=0)
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )

    parser.add_argument("--method", type=str, default=None)
    

    parser.add_argument(
        "--data_dir", type=str, default='./data/glue', help="glue base data dir."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
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
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            " should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    
    # Support enable_galore
    parser.add_argument("--enable_galore", action="store_true", help="Whether or not to use low rank optimizer.")
    # update_proj_gap
    parser.add_argument("--update_proj_gap", type=int, default=50)
    # galore_scale
    parser.add_argument("--galore_scale", type=float, default=1.0)
    # proj_type
    parser.add_argument("--proj_type", type=str, default="std")
    # lora_all_modules
    parser.add_argument("--lora_all_modules", action="store_true", help="Whether or not to use lora for all modules.")
    # eval_llama
    parser.add_argument("--eval_llama", action="store_true", help="Whether or not to evaluate llama model.")
    # low_rank_method
    parser.add_argument("--low_rank_method", type=str, default=None, help="low rank method for wandb sweep")
    
    parser.add_argument("--alpha", type=float, default=2.0)
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

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def main():
    print('Start GLUE fine-tuning!')
    args = parse_args()
    # Sending telemetry
    send_example_telemetry("run_glue_trainer", args)

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    # if args.push_to_hub:
    #     # Handle the repository creation
    #     if args.hub_model_id is None:
    #         repo_name = args.output_dir.split(os.path.sep)[-1]
    #     else:
    #         repo_name = args.hub_model_id
    #     create_repo(repo_name, exist_ok=True, token=args.hub_token)
    #     repo = Repository(args.output_dir, clone_from=repo_name)

    # Get the datasets
    if args.task_name is not None:
        # Load from local files
        data_files = {}
        train_file, validation_file = task_to_data_dir[args.task_name]
        train_file = os.path.join(args.data_dir, args.task_name, train_file)
        validation_file = os.path.join(args.data_dir, args.task_name, validation_file)
        if train_file is not None:
            data_files["train"] = train_file
        if validation_file is not None:
            data_files["validation"] = validation_file
        extension = (train_file if train_file is not None else validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        # Loading the dataset from local csv or json file
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if not args.eval_llama:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        setattr(config, 'num_labels', num_labels)
        setattr(config, 'finetuning_task', args.task_name)
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
        tokenizer.padding_side = "left"
        model = LlamaForSequenceClassification(
            config
        )

    # Load pretrained model weights
    if args.load_pretrained_model:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.load_pretrained_model}")
        checkpoint_path = os.path.join(args.load_pretrained_model, "pytorch_model.bin")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for key in checkpoint.keys():
            if key not in model.state_dict().keys():
                print(f"Key {key} not in model state dict")
        for key in model.state_dict().keys():
            if key not in checkpoint.keys():
                print(f"Key {key} not in checkpoint")
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Model successfully loaded (strict=False policy)")
        logger.info("*" * 40)

    # Set target modules for LoRA
    if not args.lora_all_modules:
        target_modules_list = ["q_proj", "v_proj"]
    else:
        print('Enabling LoRA for all modules')
        target_modules_list = ["q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "k_proj", "o_proj"]
    # Other modules for BERT-family models
    if 'bert' in args.model_name_or_path:
        if not args.lora_all_modules:
            target_modules_list = ["query"]
        else:
            print('Enabling LoRA for all modules')
            target_modules_list = ["query", "value", "key", "intermediate.dense", "output.dense"]

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Try to have some nice defaults but don't hesitate to tweak to your use case
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
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
                # Map labels to IDs
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # Rename the column to labels
                result["labels"] = examples["label"]
        return result

    with torch.no_grad():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer, 
            pad_to_multiple_of=(8 if torch.cuda.is_available() else None)
        )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    if args.method == 'lora':
        print('Using PEFT LoRA')
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            inference_mode=False,
            target_modules=target_modules_list,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.method == 'sgd':
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)    
    elif args.method == 'lion':
        optimizer = Lion(optimizer_grouped_parameters, lr=args.learning_rate) 
    elif args.method == 'cadamw':
        optimizer =CAdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    elif args.method =='adamw':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    elif args.method == 'mgup_adamw':
        optimizer = MGAdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    elif args.method == 'muon':
        optimizer = Muon(
            lr=args.learning_rate,
            wd=0.0,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    elif args.method == 'cmuon':
        optimizer = CMuon(
            lr=args.learning_rate,
            wd=0.0,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    elif args.method == 'mgup_muon':
        optimizer = MGMuon(
            lr=args.learning_rate,
            wd=0.0,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    
    elif args.method == 'lion':
        optimizer = Lion(optimizer_grouped_parameters, lr=args.learning_rate,betas=(0.95,0.98))
    elif args.method == 'clion':
        optimizer = CLion(optimizer_grouped_parameters, lr=args.learning_rate,betas=(0.95,0.98))
    elif args.method == 'mgup_lion':
        optimizer = MGLion(optimizer_grouped_parameters, lr=args.learning_rate,betas=(0.95,0.98))

    
    # Scheduler and math around the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load(args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")



     # 初始化 wandb
    if args.with_tracking and "wandb" in args.report_to:
        wandb_project = f'{args.task_name}_Roberta_base'
        wandb_run_name = f"run_{args.task_name}_{args.method}"

        if args.method in ['lbgdc','gdc'] :
            wandb_run_name = f"run_{args.task_name}_{args.method}_a{args.alpha}"
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            reinit=True  
        )
        logger.info(f"WandB initialized with project: glue_trainer, run name: {wandb_run_name}")
    else:
        logger.info("WandB not initialized.")
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # Evaluation
        model.eval()
        eval_loss = 0.0
        eval_steps = 0
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()
            eval_steps += 1
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            references = batch["labels"]
            metric.add_batch(
                predictions=predictions.cpu(),
                references=references.cpu(),
            )
        eval_metric = metric.compute()
        avg_train_loss = total_loss / len(train_dataloader)
        avg_eval_loss = eval_loss / eval_steps
        logger.info(f"Epoch {epoch}: Average training loss: {avg_train_loss:.4f}")
        logger.info(f"Epoch {epoch}: Average evaluation loss: {avg_eval_loss:.4f}")
        logger.info(f"Epoch {epoch}: Evaluation metric: {eval_metric}")


        # 记录到 wandb
        if args.with_tracking and "wandb" in args.report_to:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "eval_loss": avg_eval_loss,
                **eval_metric  # 将评估指标解包到日志中
            }, step=completed_steps)
        

    # Save final model
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Final model saved to {args.output_dir}")

    # 结束 wandb 运行
    if args.with_tracking and "wandb" in args.report_to:
        wandb.finish()

if __name__ == "__main__":
    main()