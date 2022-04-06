import sys
import json
import logging
from pathlib import Path
from argparse import Namespace
from fairseq.tasks.translation import TranslationConfig, TranslationTask

from model import build_model
from optim import LabelSmoothedCrossEntropyCriterion
from utils import try_load_checkpoint, validate

def valider(config: Namespace) -> float:
    # Logger
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO", # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "hw5.seq2seq"
    logger = logging.getLogger(proj)    
    
    # Task
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
    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )

    # Validation
    model = model.to(device=config.device)
    criterion = criterion.to(device=config.device)
    sequence_generator = task.build_generator([model], config)
    
    try_load_checkpoint(model, config, logger, name=config.ckpt_name)
    stats = validate(model, task, criterion, config, sequence_generator, logger, log_to_wandb=False)

    return stats["bleu"].score

if __name__ == "__main__":
    # load config
    config = json.loads(Path("./config.json").read_bytes())
    config = Namespace(**config)
    config.savedir = f"./checkpoints/{config.exp_name}"

    # parse args
    config.ckpt_name = str(sys.argv[1])

    bleu = valider(config)
    print(f"\n\n ===== Validation BLEU Score = {bleu} ===== \n\n")