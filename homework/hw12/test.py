import json
import random
from pathlib import Path
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.auto import tqdm

import gym

from model import ActorCritic
from utils import Memory, load_json, fix, ft, calc_cum_rewards

import wandb

def test(args: Namespace):
    # Env
    env = gym.make("LunarLander-v2")
    fix(env, args.seed)
    print("Env set.")

    # Model
    agent = ActorCritic(hidden_dim=args.hidden)
    agent.load_state_dict(torch.load(f"{args.save_dir}/{args.exp_name}/best_model.pth"))
    print("Model loaded.")

    agent.eval()
    NUM_OF_TEST = 5 # Do not revise this !!!
    test_total_reward = []
    action_list = []
    for _ in range(NUM_OF_TEST):
        actions = []
        state = env.reset()

        total_reward = 0

        done = False
        while not done:
            _, action, _ = agent(ft(state))
            actions.append(action)
            state, reward, done, _ = env.step(action)

            total_reward += reward
            
        print(total_reward)
        test_total_reward.append(total_reward)

        action_list.append(actions) # save the result of testing 

    print(f"===== Average total reward: {np.mean(test_total_reward)} =====")

    PATH = f"./{args.save_dir}/{args.exp_name}/action_list.npy" # Can be modified into the name or path you want
    np.save(PATH, np.array(action_list))
    return float(np.mean(test_total_reward))

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str
    )

    cmd_args = parser.parse_args()
    return cmd_args

if __name__ == "__main__":
    cmd_args = parse_args()

    config = load_json(file=cmd_args.config_file)
    args = Namespace(**config)

    test_total_reward = test(args)
    print(type(test_total_reward), test_total_reward)