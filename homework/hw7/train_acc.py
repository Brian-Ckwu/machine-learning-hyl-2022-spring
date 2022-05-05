from tqdm.auto import tqdm
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast
from accelerate import Accelerator

from data import read_data, QA_Dataset
from utils import same_seeds, load_config, evaluate

transformers.logging.set_verbosity_error()

def trainer(args: Namespace):
    # Fix random seed for reproducibility
    same_seeds(args.seed)

    # Data
    # load
    train_questions, train_paragraphs = read_data(args.train_path)
    valid_questions, valid_paragraphs = read_data(args.valid_path)
    print("Data loaded.")

    # tokenize
    tokenizer = BertTokenizerFast.from_pretrained(args.encoder)
    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
    valid_questions_tokenized = tokenizer([valid_question["question_text"] for valid_question in valid_questions], add_special_tokens=False)
    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    valid_paragraphs_tokenized = tokenizer(valid_paragraphs, add_special_tokens=False)
    print("Data tokenized.")

    # dataset
    train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    valid_set = QA_Dataset("valid", valid_questions, valid_questions_tokenized, valid_paragraphs_tokenized)    

    # dataloader
    train_loader = DataLoader(train_set, batch_size=args.bs // args.grad_accum_steps, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, pin_memory=True)   
    print("Dataset & Dataloader constructed.")

    # Model
    model = BertForQuestionAnswering.from_pretrained(args.encoder).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    print("Model prepraed.")

    # Optimization
    total_train_steps = int(len(train_loader) / args.grad_accum_steps * args.nepochs)
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer, 
        num_training_steps=total_train_steps,
        num_warmup_steps=warmup_steps
    )
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    model.train()
    print(f"Start Training with Accelerator and Scheduler. Total training steps = {total_train_steps}; warmup steps = {warmup_steps}")

    best_valid_acc = 0
    for epoch in range(args.nepochs):
        loader_idx = 0
        step = 0
        # train_loss = 0
        # train_acc = 0

        for data in tqdm(train_loader):
            model.train()
            # Load all data into GPU
            data = [i.to(args.device) for i in data]
            
            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
            loss = output.loss / args.grad_accum_steps
            accelerator.backward(loss)

            # NOTE: Choose the most probable start position / end position
            # start_index = torch.argmax(output.start_logits, dim=1)
            # end_index = torch.argmax(output.end_logits, dim=1)
            
            # NOTE: Prediction is correct only if both start_index and end_index are correct
            # train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            # train_loss += loss
            
            if (loader_idx % args.grad_accum_steps == args.grad_accum_steps - 1) or (step == len(train_loader) - 1):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                # TODO: log train loss

            ##### TODO: Apply linear learning rate decay #####

                # Print validation accuracy over past logging step
                if step % args.logsteps == 0:
                    # Evaluation
                    print("Evaluating Validation Set ...")
                    model.eval()
                    with torch.no_grad():
                        valid_acc = 0
                        for i, data in enumerate(tqdm(valid_loader)):
                            output = model(input_ids=data[0].squeeze(dim=0).to(args.device), token_type_ids=data[1].squeeze(dim=0).to(args.device),
                                attention_mask=data[2].squeeze(dim=0).to(args.device))
                            # prediction is correct only if answer text exactly matches
                            valid_acc += evaluate(data, output, tokenizer) == valid_questions[i]["answer_text"]
                        valid_acc /= len(valid_loader)
                        print(f"Validation | Epoch {epoch + 1} | Step {step} | acc = {valid_acc:.3f}")
                    
                    # Save best model
                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        # Save a model and its configuration file to the directory 「saved_model」 
                        # i.e. there are two files under the direcory 「saved_model」: 「pytorch_model.bin」 and 「config.json」
                        # Saved model can be re-loaded using 「model = BertForQuestionAnswering.from_pretrained("saved_model")」
                        print("Saving Best Model ...")
                        model.save_pretrained(args.model_save_dir)
                        (Path(args.model_save_dir) / "best_acc.txt").write_text(str(best_valid_acc))

            loader_idx += 1   

    return best_valid_acc 

if __name__ == "__main__":
    config = load_config("./config.json")
    args = Namespace(**config)

    best_valid_acc = trainer(args)
    print(f"\nFinish training. Best validation accuracy = {best_valid_acc:.4f}")