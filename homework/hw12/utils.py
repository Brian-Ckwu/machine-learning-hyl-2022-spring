import json
from typing import List
from pathlib import Path

import random
import numpy as np
import torch

class Memory(object):
    # initialize container for memorizing training values
    def __init__(self):
        self.dones = list()
        self.values = list()
        self.rewards = list()
        self.log_probs = list()
    
    def record(self, done, value, reward, log_prob):
        self.dones.append(done)
        self.values.append(value)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def clear(self):
        self.dones.clear()
        self.values.clear()
        self.rewards.clear()
        self.log_probs.clear()

def load_json(file: str):
    return json.loads(Path(file).read_bytes())

def fix(env, seed: int):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ft(x) -> torch.Tensor:
    return torch.FloatTensor(x)

def calc_cum_rewards(rewards: List[float], gamma: float):
    cum_rewards = torch.zeros(size=(len(rewards),))
    cum_reward = 0
    for i in reversed(range(len(rewards))):
        reward = rewards[i]
        cum_reward = reward + gamma * cum_reward
        cum_rewards[i] = cum_reward
    
    return cum_rewards