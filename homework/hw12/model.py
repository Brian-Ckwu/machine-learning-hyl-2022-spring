import torch
import torch.nn as nn

from torch.distributions import Categorical

class ActorCritic(nn.Module):
    # initialize model parameters
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.actor_layer = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.Softmax(dim=-1)
        )
        self.critic_layer = nn.Linear(hidden_dim, 1)

    # evaluate state and sample action
    # NOTE: currently reinforce with baseline
    def forward(self, state: torch.FloatTensor) -> tuple:
        # shared hidden state
        hidden = self.shared_net(state)

        # critic
        value = self.critic_layer(hidden)

        # actor
        action_probs = self.actor_layer(hidden)
        action_dist = Categorical(probs=action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(value=action)

        return value, action.item(), log_prob
    
    def calc_loss(self, values: torch.FloatTensor, cum_rewards: torch.FloatTensor, log_probs: torch.FloatTensor) -> torch.FloatTensor:
        # calculate relative rewards
        advantages = cum_rewards - values
        # calculate losses
        critic_loss = advantages.pow(2).mean() # TODO: change to smooth-l1-loss?
        actor_loss = (-log_probs * advantages.detach()).mean() # NOTE: rewards should not attach gradients

        total_loss = critic_loss + actor_loss
        return total_loss