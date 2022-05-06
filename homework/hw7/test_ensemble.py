import re
from tqdm.auto import tqdm
from pathlib import Path
from argparse import Namespace

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import BertForQuestionAnswering, BertTokenizerFast
from accelerate import Accelerator

from data import read_data, QA_Dataset
from utils import same_seeds, load_config, evaluate

transformers.logging.set_verbosity_error()

def make_pred_file(args: Namespace):
    same_seeds(args.seed)
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)

    # Data
    test_questions, test_paragraphs = read_data(args.test_path)
    print("Data loaded.")

    tokenizer = BertTokenizerFast.from_pretrained(args.encoder)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)
    print("Data tokenized.")

    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
    print("Dataset & Dataloader constructed.")

    # Model
    models = list()
    for remainder in range(10):
        print(f"Loading model {remainder}...")
        args.model_save_dir = re.sub(pattern=r"remainder\-\d", repl=f"remainder-{remainder}", string=args.model_save_dir)
        print(args.model_save_dir)
        model = BertForQuestionAnswering.from_pretrained(args.model_save_dir, local_files_only=True).to(args.device)
        model = accelerator.prepare(model)
        models.append(model)
    print("Models loded.")

    # Make Prediction
    print("Making prediction...")
    result = list()
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            outputs = list()
            for model in models:
                output = model(input_ids=data[0].squeeze(dim=0).to(args.device), token_type_ids=data[1].squeeze(dim=0).to(args.device), attention_mask=data[2].squeeze(dim=0).to(args.device))
                outputs.append(output)
            
            summed_output = sum_output(outputs)
            result.append(evaluate(data, summed_output, tokenizer, args.max_ans_length, args.n_best))

    print("Saving prediction...")
    pred_path = Path(args.ensemble_save_dir) / "prediction.csv"
    with open(pred_path, 'w', encoding="utf-8") as f:	
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    return pred_path

def sum_output(outputs):
    summed_output = outputs[0]
    for output in outputs[1:]:
        summed_output.start_logits += output.start_logits
        summed_output.end_logits += output.end_logits
    return summed_output

if __name__ == "__main__":
    config = load_config("./config.json")
    args = Namespace(**config)

    pred_path = make_pred_file(args)
    print(f"Prediction file saved to {pred_path}")