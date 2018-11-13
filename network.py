import time
import gym
import gym_minigrid
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions.categorical import Categorical
from itertools import count
from copy import deepcopy
import utils
import ewc

# A simple, memoryless MLP agent. Last layer are logits (scores for
# which higher values represent preferred actions).

###################################################################################
#
# Some useful instructions:
#
# 1. The creation of the data folder, of the model folder and so on are automatic,
#    all you have to remember is to change the 'task' and 'env_name' parameters; if
#    you forget that, you'll probably have to re-train everything in order to have
#    data that make sense.
#
# 2. If you delete a model, make sure to delete also its related data folder, or
#    you'll have mixed data from both the previous and the actual model. Don't
#    panic: as you create a new agent, the program will automatically create a new
#    data folder.
#
###################################################################################

class Policy(nn.Module):
    def __init__(self, obs_size, act_size, inner_size, **kwargs):
        super(Policy, self).__init__(**kwargs)
        self.affine1 = nn.Linear(obs_size, inner_size)
        self.affine2 = nn.Linear(inner_size, act_size)

        self.saved_log_probs = []
        self.rewards = []

        self.FIM = {}

    def forward(self, x):
        x = x.view(-1, 7*7)
        x = F.relu(self.affine1(x))
        act_probs = self.affine2(x).clamp(-1000.0, +1000.0)
        return act_probs

    def set_FIM(self, FIM):
        self.FIM = FIM

# Function that, given a policy network and a state selects a random
# action according to the probabilities output by final layer.
def select_action(policy, state):
    probs = policy.forward(state)
    dist = Categorical(logits=probs)
    action = dist.sample()
    return action

# Utility function. The MiniGrid gym environment uses 3 channels as
# state, but for this we only use the first channel: represents all
# objects (including goal) with integers. This function just strips
# out the first channel and returns it.
def state_filter(state):
    return torch.from_numpy(state['images'][:,:,0]).float()

# Function to compute discounted rewards after a complete episode.
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        discounted_rewards.append(running)
    return list(reversed(discounted_rewards))

# The function that runs the simulation for a specified length. The
# nice thing about the MiniGrid environment is that the game never
# ends. After achieving the goal, the game resets. Kind of like
# Sisyphus...
def run_episode(env, policy, length, gamma=0.99):
    # Restart the MiniGrid environment.
    state = state_filter(env.reset())

    # We need to keep a record of states, actions, and the
    # instantaneous rewards.
    states = [state]
    actions = []
    rewards = []

    # Run for desired episode length.
    for step in range(length):
        # Get action from policy net based on current state.
        action = select_action(policy, state)

        # Simulate one step, get new state and instantaneous reward.
        state, reward, done, _ = env.step(action)
        state = state_filter(state)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        if done:
            break

    # Finished with episode, compute loss per step.
    discounted_rewards = compute_discounted_rewards(rewards, gamma)

    # Return the sequence of states, actions, and the corresponding rewards.
    return (states, actions, discounted_rewards)



