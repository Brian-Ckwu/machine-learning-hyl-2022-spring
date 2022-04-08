import sys
import pdb # python debugger
import pprint # pretty print
import logging
import os
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import Namespace
from fairseq import utils
from fairseq.tasks.translation import TranslationConfig, TranslationTask


from data import load_data_iterator
from model import build_model
from optim import LabelSmoothedCrossEntropyCriterion, NoamOpt
from utils import *
from preprocess import clean_corpus

def trainer(config: Namespace):
    set_seed(config.myseed)

    # Logging
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO", # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)

    # Data
    task_cfg = TranslationConfig(
        data=config.datadir,
        source_lang=config.source_lang,
        target_lang=config.target_lang,
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
    task = TranslationTask.setup_task(task_cfg)
    logger.info("loading data for epoch 1")
    task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
    task.load_dataset(split="valid", epoch=1)

    # Model
    arch_args = Namespace(**json.loads(Path(config.model_arch_path).read_bytes()))
    model = build_model(arch_args, task)
    logger.info(model)

    # Optimization
    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )
    optimizer = NoamOpt(
        model_size=arch_args.encoder_embed_dim, 
        factor=config.lr_factor, 
        warmup=config.lr_warmup, 
        optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001)
    )

    # Training
    sequence_generator = task.build_generator([model], config)
    model = model.to(device=config.device)
    criterion = criterion.to(device=config.device)

    log_info(logger, task, model, criterion, optimizer, config)

    epoch_itr = load_data_iterator(task, "train", config.myseed, config.start_epoch, config.max_tokens, config.num_workers)
    try_load_checkpoint(model, config, logger, optimizer, name=config.resume)

    bleus = list() # save gnorms
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config, logger, config.accum_steps)
        stats = validate_and_save(model, task, criterion, config, optimizer, epoch_itr.epoch, sequence_generator, logger)
        bleus.append(stats["bleu"].score)
        logger.info("end of epoch {}".format(epoch_itr.epoch))    
        epoch_itr = load_data_iterator(task, "train", config.myseed, epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)
        # write performance
        (Path(config.savedir) / "ckpt_bleu.json").write_text(data=json.dumps(bleus))

    return bleus


if __name__ == "__main__":
    # load config
    config = json.loads(Path("./config.json").read_bytes())
    config = Namespace(**config)
    config.savedir = f"./checkpoints/{config.exp_name}"

    # training
    bleus = trainer(config)
    print(bleus)