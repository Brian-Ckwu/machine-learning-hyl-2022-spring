import json
import random
from tqdm.auto import tqdm
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset 

import transformers
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast

from data import read_data, QA_Dataset
from utils import same_seeds, load_config, evaluate

transformers.logging.set_verbosity_error()

def validate(args: Namespace):
    same_seeds(args.seed)

    # Data
    valid_questions, valid_paragraphs = read_data(args.valid_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.encoder)

    valid_questions_tokenized = tokenizer([valid_question["question_text"] for valid_question in valid_questions], add_special_tokens=False)
    valid_paragraphs_tokenized = tokenizer(valid_paragraphs, add_special_tokens=False)

    valid_set = QA_Dataset("valid", valid_questions, valid_questions_tokenized, valid_paragraphs_tokenized)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True)

    # Model
    model = BertForQuestionAnswering.from_pretrained(args.model_save_dir, local_files_only=True).to(args.device)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        ncorrect = 0
        for i, data in enumerate(tqdm(valid_loader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(args.device), token_type_ids=data[1].squeeze(dim=0).to(args.device),
                attention_mask=data[2].squeeze(dim=0).to(args.device))
            # prediction is correct only if answer text exactly matches
            ncorrect += evaluate(data, output, tokenizer) == valid_questions[i]["answer_text"]
    
    valid_acc = ncorrect / len(valid_loader)
    return valid_acc

if __name__ == "__main__":
    config = load_config("./config.json")
    args = Namespace(**config)

    valid_acc = validate(args)
    print(f"Validation Accuracy = {valid_acc:.4f}")