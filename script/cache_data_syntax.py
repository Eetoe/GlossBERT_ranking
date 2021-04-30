import argparse
import logging
import os
import random

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils.dataset_syntax import load_dataset
from utils.model_syntax import BERT_MODELS, get_model_and_tokenizer

import spacy
from spacy.symbols import ORTH
import regex as re

"""
Script modified from the run_model.py script.
The reason to create this script was for caching the data, a process, which seemed slow on GPUs.
The caching seemed faster on local CPU, thus this script was made to do the caching before the training.

Some of the names for the args are a legacy from the run_model.py script, as the utils function are
tailor-made to take the args from run_model.py.
"""

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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
        help="The output directory where the model will be loaded into. Deleted by the end of the script.",
    )

    # Other parameters
    parser.add_argument(
        "--train_path",
        default="",
        type=str,
        help="Path to the dataset to be cached (.csv file). Name is a legacy.",
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
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
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
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    elif not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:  # Make output dir if it doesn't exist
        os.makedirs(args.output_dir)

    # ====== Setup CUDA, GPU & distributed training ======
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
    logger.info("Caching parameters %s", args)

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

    logger.info("Loading BERT model...\n")
    model, tokenizer = get_model_and_tokenizer(args)
    logger.info("\n\nBERT model loaded!\n")

    # When loading the BERT model, a directory is created with it's pos and dep vocab
    # This is a legacy from the run_model.py being made for also caching the data
    # This directory is deleted here because if it isn't empty, the training will throw a warning
    # The directroy will be created again during training
    if len([name for name in os.listdir(args.output_dir) if os.path.isfile(name)]) < 3:
        os.remove(args.output_dir+"/dep_vocab.txt")
        os.remove(args.output_dir + "/pos_vocab.txt")
        os.rmdir(args.output_dir)

    logger.info("\n\nStart caching dataset...")
    train_dataset = load_dataset(args, args.train_path, tokenizer, args.max_seq_length, spacy_model)
    logger.info("\n\nDataset cached!")


if __name__ == "__main__":
    main()
