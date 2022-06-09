import json
import random
from pathlib import Path
from argparse import Namespace

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

def evaluate(args: Namespace, agent: ActorCritic):
    # Env
    env = gym.make("LunarLander-v2")
    fix(env, args.seed)
    # print("Env set.")

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
            
        # print(total_reward)
        test_total_reward.append(total_reward)

        action_list.append(actions) # save the result of testing 

    # print(f"===== Average total reward: {np.mean(test_total_reward)} =====")
    return float(np.mean(test_total_reward)), action_list

def main(args: Namespace):
    # Config
    save_path = Path(f"{args.save_dir}/{args.exp_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "config.json").write_text(json.dumps(vars(args)))

    # Env
    env = gym.make("LunarLander-v2")
    fix(env, args.seed)

    initial_state = env.reset()
    random_action = env.action_space.sample()
    observation, reward, done, info = env.step(random_action)
    print(f"Initial state: {initial_state}") # to check it's the correct seed

    # Model
    if args.model == "shared":
        agent = ActorCritic(hidden_dim=args.hidden)
    
    optimizer = getattr(torch.optim, args.optimizer)(agent.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes // args.update_episodes, eta_min=args.min_lr)

    # Optimization
    memory = Memory()
    ep_total_rewards, ep_final_rewards = list(), list()
    running_total_rewards, running_final_rewards = list(), list()
    grad_norms = list()
    best_running_reward = -1000
    best_test_reward = -1000

    values, cum_rewards, log_probs = [torch.FloatTensor()] * 3

    pbar = tqdm(range(args.episodes))
    agent.train()
    # NOTE: Current Implementation is Monte-Carlo (Reinforce with baseline)
    for episode in pbar:
        state = env.reset()
        ep_total_reward = 0
        
        while True:
            # interact with env.
            value, action, log_prob = agent(ft(state))
            next_state, reward, done, _ = env.step(action)
            # record training values
            memory.record(done, value, reward, log_prob)
            ep_total_reward += reward
            # update state
            state = next_state

            if done:
                ep_total_rewards.append(ep_total_reward)
                ep_final_rewards.append(reward)
                wandb.log({"episode_total_reward": ep_total_reward, "episode_final_reward": reward}, step=episode)
                break
        
        # prepare values for calculating loss
        if args.normalize:
            rewards = np.array(memory.rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + args.std_eps)
        else:
            rewards = memory.rewards
        values = torch.cat(tensors=(values, torch.cat(memory.values, dim=0)))
        cum_rewards = torch.cat(tensors=(cum_rewards, calc_cum_rewards(rewards, gamma=args.gamma)))
        log_probs = torch.cat(tensors=(log_probs,torch.stack(memory.log_probs)))

        # update model parameters every k episode
        if (episode + 1) % args.update_episodes == 0:
            loss = agent.calc_loss(values, cum_rewards, log_probs)
            loss.backward()
            if args.clip_grad_norm:
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_norm)
                grad_norms.append(grad_norm.item())
                wandb.log({"grad_norm": grad_norm}, step=episode)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # reset containers
            values, cum_rewards, log_probs = [torch.FloatTensor()] * 3
        
        # logging
        running_total_reward = sum(ep_total_rewards[-args.log_episodes:]) / len(ep_total_rewards[-args.log_episodes:])
        running_final_reward = sum(ep_final_rewards[-args.log_episodes:]) / len(ep_final_rewards[-args.log_episodes:])
        running_total_rewards.append(running_total_reward)
        running_final_rewards.append(running_final_reward)
        pbar.set_description(f"Total: {running_total_reward:4.1f}; Final: {running_final_reward:4.1f} (lr = {optimizer.param_groups[0]['lr']:.5f})")
        wandb.log({"running_total_reward": running_total_reward, "running_final_reward": running_final_reward, "lr": optimizer.param_groups[0]['lr']}, step=episode)

        # save the best model
        if running_total_reward > best_running_reward:
            best_running_reward = running_total_reward
            torch.save(obj=agent.state_dict(), f=save_path / "best_model.pth")
            (save_path / "best_running_reward.txt").write_text(str(best_running_reward))
            wandb.log({"best_running_reward": best_running_reward}, step=episode)
        
        if args.evaluate_during_training:
            test_avg_reward, action_list = evaluate(args, agent)
            if test_avg_reward > best_test_reward:
                best_test_reward = test_avg_reward
                torch.save(obj=agent.state_dict(), f=save_path / "best_test_model.pth") # save model
                np.save(f"./{args.save_dir}/{args.exp_name}/action_list-{int(test_avg_reward)}.npy", np.array(action_list)) # save action list
                wandb.log({"best_test_reward": best_test_reward})

        # clear memory
        memory.clear()

    wandb.finish(exit_code=0)
    return best_running_reward

if __name__ == "__main__":
    config = load_json("./config.json")
    args = Namespace(**config)

    wandb.init(
        project=args.project_name,
        name=args.exp_name,
        config=vars(args)
    )

    best_running_reward = main(args)
    print(f"Best running reward (avg {args.log_episodes} episodes) = {best_running_reward}")