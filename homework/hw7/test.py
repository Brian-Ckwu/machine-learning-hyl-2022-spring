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
    tokenizer = BertTokenizerFast.from_pretrained(args.encoder)

    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    # Model
    model = BertForQuestionAnswering.from_pretrained(args.model_save_dir, local_files_only=True).to(args.device)
    model = accelerator.prepare(model)

    # Make Prediction
    print("Making prediction...")
    result = list()
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(args.device), token_type_ids=data[1].squeeze(dim=0).to(args.device),
                        attention_mask=data[2].squeeze(dim=0).to(args.device))
            result.append(evaluate(data, output, tokenizer))

    print("Saving prediction...")
    pred_path = Path(args.model_save_dir) / "prediction.csv"
    with open(pred_path, 'w') as f:	
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # Replace commas in answers with empty strings (since csv is separated by comma)
            # Answers in kaggle are processed in the same way
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    return pred_path

if __name__ == "__main__":
    config = load_config("./config.json")
    args = Namespace(**config)

    pred_path = make_pred_file(args)
    print(f"Prediction file saved to {pred_path}")