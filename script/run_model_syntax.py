# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Modified from the script at https://github.com/huggingface/transformers/blob/master/examples/run_glue.py """

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils.dataset_syntax import load_dataset, write_predictions, \
    collate_batch, collate_batch_pos_dep, collate_batch_pos, collate_batch_dep
from utils.model_syntax import BERT_MODELS, BertWSDArgs, get_model_and_tokenizer, forward_gloss_selection, \
    BertTokenizerArgs, BertConfigSyntax

import spacy
from spacy.symbols import ORTH
import regex as re

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, model, tokenizer, train_dataloader, eval_during_training=False):
    """ Fine-tune the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        # t_total is the training length calculated above.
    )

    # Check if saved optimizer or scheduler states exist
    if (
            os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num samples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.per_gpu_train_batch_size
        * max(1, args.n_gpu)
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, args.num_train_epochs, desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            loss = forward_gloss_selection(args, model, batch)[0]
            #logger.info("forward gloss selection done")
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            #logger.info("about to calculate training loss")
            tr_loss += loss.item()  # .item() is to get a number out of the tensor instead of the tensor itself.
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()  # removes accumulated gradients to update correctly at current step.
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and eval_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        logs["eval_loss"] = evaluate(args, model, tokenizer, global_step)

                    loss_scalar = (
                                          tr_loss - logging_loss) / args.logging_steps  # average loss of steps since last logging
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)

                    with open(os.path.join(args.output_dir, "train_log.txt"), 'a+') as f:
                        print(json.dumps({**logs, **{"step": global_step}}), file=f)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, training_args, model, tokenizer, suffix=None):
    # Defined in


    # ====== Set up spacy model ======
    if training_args.use_pos_tags or training_args.use_dependencies:
        assert args.spacy_model in ["en_core_web_trf", "en_core_web_sm"], \
            'The spacy model has to be either "en_core_web_trf" or "en_core_web_sm". '
        logger.info("Loading SpaCy model...")
        spacy_model = spacy.load(args.spacy_model)
        add_tgt = [{ORTH: "[TGT]"}]
        spacy_model.tokenizer.add_special_case("[TGT]", add_tgt)
        logger.info("SpaCy model loaded!\n")
    else:
        spacy_model = None

    eval_dataset = load_dataset(training_args, args.eval_path, tokenizer, training_args.max_seq_length,
                                spacy_model=spacy_model)
    args.eval_batch_size = args.eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    if training_args.use_pos_tags and training_args.use_dependencies:
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_batch_pos_dep)
    elif training_args.use_pos_tags and not training_args.use_dependencies:
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_batch_pos)
    elif not training_args.use_pos_tags and training_args.use_dependencies:
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_batch_dep)
    else:
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=collate_batch)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    predictions = []
    for batches in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            loss, logits_list = forward_gloss_selection(training_args, model, batches)

        eval_loss += loss
        # argmax with dimension returns the position of the highest value for the dimension for each unit in that dimension, e.g., rows or columns.
        # dim = 0 is columns, 1 is rows, -1 might be the whole array.
        predictions.extend([torch.argmax(logits, dim=-1).item() for logits in logits_list])
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    write_predictions(args.output_dir, args.eval_path, predictions, suffix)

    return eval_loss.item()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(BERT_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written."
             " If auto_create, the script will automatically make a name based on the parameters.",
    )

    # Other parameters
    parser.add_argument(
        "--train_path",
        default="",
        type=str,
        help="Path to training dataset (.csv file).",
    )
    parser.add_argument(
        "--eval_path",
        default="",
        type=str,
        help="Path to evaluation dataset (.csv file).",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Cache directory to store the pre-trained models downloaded from s3.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(  # Epsilon is a very small parameter used to avoid division by 0
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps",
        default=100,
        type=int,
        help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        default=2000,
        type=int,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training sets"
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed for initialization"
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training on train set."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run evaluation on dev/test set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step."
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--server_ip",
        default="",
        type=str,
        help="For distant debugging."
    )
    parser.add_argument(
        "--server_port",
        default="",
        type=str,
        help="For distant debugging."
    )
    # ====== ADD syntax args ======
    parser.add_argument(
        "--use_pos_tags",
        action="store_true",
        help="Whether to use pos information in the model.",
    )
    parser.add_argument(
        "--use_dependencies",
        action="store_true",
        help="Whether to use grammatical dependency information in the model.",
    )
    parser.add_argument(
        "--zero_syntax_for_special_tokens",
        action="store_true",
        help="Whether to multiply the pos and dep tensors with 0 for [CLS], [SEP] and [TGT] tokens.",
    )
    parser.add_argument(
        "--spacy_model",
        default="en_core_web_trf",
        help="Which spacy model to use. The base model is en_core_web_sm; the transformer model is en_core_web_trf."
    )
    parser.add_argument(
        "--use_gloss_extensions",
        action="store_true",
        help="Whether to use gloss extension, i.e. adding the target word to the gloss. NB: only used along with POS.",
    )
    parser.add_argument(
        "--gloss_extensions_w_tgt",
        action="store_true",
        help="Whether to add [TGT] tokens around the target word in the gloss extension.",
    )
    args = parser.parse_args()

    if not args.use_gloss_extensions and args.gloss_extensions_w_tgt:
        raise ValueError("To add [TGT] tokens to the gloss extensions, please turn on gloss extensions first.")

    # ====== Create output directory name ======
    if args.output_dir == "auto_create":
        if re.search("/", args.model_name_or_path):
            auto_created_name = args.model_name_or_path.split("/")[-1]
        else:
            auto_created_name = args.model_name_or_path

        if args.use_pos_tags:
            auto_created_name += "-pos"
        if args.use_dependencies:
            auto_created_name += "-dep"
        if args.zero_syntax_for_special_tokens:
            auto_created_name += "-no_syntax_for_special"
        if args.use_gloss_extensions:
            auto_created_name += "-glosses_extended"
        if args.gloss_extensions_w_tgt:
            auto_created_name += "_w_tgt"
        if re.search("-augmented", args.train_path):
            auto_created_name += "-augmented"
        if re.search("max_num_gloss=(\d)+", args.train_path):
            auto_created_name += re.search(r"-max_num_gloss=(\d)+", args.train_path)[0]
        auto_created_name += "-batch_size=" + str(args.per_gpu_train_batch_size)
        auto_created_name += "-lr=" + str(args.learning_rate)
        auto_created_name = "model/"+auto_created_name
        args.output_dir = auto_created_name

    # ====== Makes sure not to overwrite stuff in output dir unless you want to ======
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    elif not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:  # Make output dir if it doesn't exist
        os.makedirs(args.output_dir)

    # ====== Setup distant debugging if needed ======
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        # Use cuda if available and no_cuda is False, else use cpu
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        # asctime is simply human readable time stamps.
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,  # WARN is a deprecated version of warning
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)

    # ====== Set up spacy model ======
    if args.use_pos_tags or args.use_dependencies:
        assert args.spacy_model in ["en_core_web_trf", "en_core_web_sm"], \
            'The spacy model has to be either "en_core_web_trf" or "en_core_web_sm". '
        logger.info("Loading SpaCy model...\n")
        spacy_model = spacy.load(args.spacy_model)
        add_tgt = [{ORTH: "[TGT]"}]
        spacy_model.tokenizer.add_special_case("[TGT]", add_tgt)
        logger.info("SpaCy model loaded!\n")
    else:
        spacy_model = None

    # Set seed
    set_seed(args)

    # Training
    if args.do_train:
        # Load pretrained model and tokenizer
        model, tokenizer = get_model_and_tokenizer(args)

        # Calculate batch size for data loader
        batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

        def _get_dataloader(_train_dataset, _collate_fn):
            # \ by itself joins two lines together using the logic of the first line
            train_sampler = RandomSampler(_train_dataset) if args.local_rank == -1 \
                else DistributedSampler(_train_dataset)

            return DataLoader(
                _train_dataset,
                sampler=train_sampler,
                batch_size=batch_size,
                collate_fn=_collate_fn
            )

        # fine-tune on gloss selection task
        logger.info("\nTraining...")
        train_dataset = load_dataset(args, args.train_path, tokenizer, args.max_seq_length, spacy_model)
        # Load the appropriate collate function depending on the use of pos and dep
        if args.use_pos_tags and args.use_dependencies:
            train_dataloader = _get_dataloader(train_dataset, collate_batch_pos_dep)
        elif args.use_pos_tags and not args.use_dependencies:
            train_dataloader = _get_dataloader(train_dataset, collate_batch_pos)
        elif not args.use_pos_tags and args.use_dependencies:
            train_dataloader = _get_dataloader(train_dataset, collate_batch_dep)
        else:
            train_dataloader = _get_dataloader(train_dataset, collate_batch)

        global_step, tr_loss = train(args, model, tokenizer, train_dataloader, args.evaluate_during_training)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        logger.info("\nStart evaluation!\n")
        # ====== Load training args ======
        training_args = torch.load(args.model_name_or_path + "/training_args.bin")

        # ====== Load fine-tuned model, its configguration and its tokenizer ======
        # Load tokenizer
        if training_args.use_pos_tags:
            pos_vocab_path = training_args.output_dir + "/pos_vocab.txt"
        if training_args.use_dependencies:
            dep_vocab_path = training_args.output_dir + "/dep_vocab.txt"

        if training_args.use_pos_tags and training_args.use_dependencies:
            tokenizer = BertTokenizerArgs.from_pretrained(args.output_dir,
                                                          training_args,
                                                          pos_vocab_file=pos_vocab_path,
                                                          dep_vocab_file=dep_vocab_path)
        elif training_args.use_pos_tags and not training_args.use_dependencies:
            tokenizer = BertTokenizerArgs.from_pretrained(args.output_dir,
                                                          training_args,
                                                          pos_vocab_file=pos_vocab_path)
        elif not training_args.use_pos_tags and training_args.use_dependencies:
            tokenizer = BertTokenizerArgs.from_pretrained(args.output_dir,
                                                          training_args,
                                                          dep_vocab_file=dep_vocab_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(args.output_dir)

        assert "[TGT]" in tokenizer.additional_special_tokens
        logger.info(f"Vocab size in tokenizer: {len(tokenizer)}")
        logger.info("Tokenizer loaded!")

        # Load configuration
        training_config = BertConfigSyntax.from_pretrained(
            args.model_name_or_path,
            num_labels=2,
            cache_dir=args.cache_dir if args.cache_dir else None
        )
        logger.info(f"Vocab size in config: {training_config.vocab_size}")
        logger.info("Config loaded!")

        # Load model
        model = BertWSDArgs.from_pretrained(
            args.model_name_or_path,
            training_args,
            config=training_config
        )
        logger.info("Model loaded!")

        # ====== Send model to device and start evaluating ======
        model.to(args.device)

        # Note that both args and training args are needed.
        eval_loss = evaluate(args, training_args, model=model, tokenizer=tokenizer)
        print(f"Evaluation loss: {eval_loss}")


if __name__ == "__main__":
    main()
