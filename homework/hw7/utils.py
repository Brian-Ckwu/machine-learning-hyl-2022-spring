import json
import numpy as np
import random
from pathlib import Path

import torch

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(file_path: str):
    return json.loads(Path(file_path).read_bytes())

def evaluate(data, output, tokenizer, max_ans_length, n_best):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    for k in range(num_of_windows):
        start_indices = torch.argsort(output.start_logits[k], dim=0)[-n_best:].tolist()
        end_indices = torch.argsort(output.end_logits[k], dim=0)[-n_best:].tolist()

        for start_index in start_indices:
            for end_index in end_indices:
                if (end_index >= start_index) and (end_index - start_index < max_ans_length):
                    start_prob = output.start_logits[k][start_index].item()
                    end_prob = output.end_logits[k][end_index].item()
                    prob = start_prob + end_prob
                    if (prob > max_prob):
                        max_prob = prob
                        # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
                        answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')