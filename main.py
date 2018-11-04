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
    return torch.from_numpy(state['image'][:,:,0]).float()

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

###### The main loop.
if __name__ == '__main__':
    ###### Some configuration variables.
    episode_len = 50                      # Length of each game.
    obs_size = 7*7                        # MiniGrid uses a 7x7 window of visibility.
    act_size = 7                          # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
    inner_size = 64                       # Number of neurons in two hidden layers.
    lr = 0.001                            # Adam learning rate
    avg_reward = 0.0                      # For tracking average regard per episode.
    env_name = 'MiniGrid-Empty-6x6-v0'    # Size of the grid
    task = "task-1"                       # Task name for saving data-task-1 on the right folder
    first_write_flag = True               # Need this due to a weird behavior of the library
    training = True                      # If set to False, optimizer won't run (and the net won't learn)
    plot = False                          # If true, plots all the important stuff
    need_FIM = True                      # Avoid the FIM calculus if not required
    goal_pos = 1                          # Change the goal square position

    # Check whether the data directory exists and, if not, create it with all the necessary stuff.
    if not os.path.exists("data-{task}/".format(task=task)):
        print("Task 2 data directory created.")
        os.makedirs("data-{task}/".format(task=task))
    if not os.path.exists("data-{task}/{env_name}/".format(task=task, env_name=env_name)):
        os.makedirs("data-{task}/{env_name}/".format(task=task, env_name=env_name))

    output_reward = open("data-{task}/{env_name}/reward.txt".format(task=task, env_name=env_name), 'a+')
    output_avg = open("data-{task}/{env_name}/avg_reward.txt".format(task=task, env_name=env_name), 'a+')
    output_loss = open("data-{task}/{env_name}/loss.txt".format(task=task, env_name=env_name), 'a+')

    # Setup OpenAI Gym environment for guessing game.
    env = gym.make(env_name)
    if goal_pos == 2:
        env.set_posX(3)
        env.set_posY(4)

    # Check the model directory
    last_checkpoint = utils.search_last_model('torch_models/', env_name, task)

    # Instantiate a policy network
    policy = Policy(obs_size=obs_size, act_size=act_size, inner_size=inner_size)

    # If there's a previous checkpoint, load this instead of using a new one.
    if os.listdir('torch_models/{env}-{task}/'.format(env=env_name, task=task)):
        policy.load_state_dict(torch.load("torch_models/{env_name}-{task}/model-{env_name}-{step}.pth".format(
            env_name=env_name, step=last_checkpoint, task=task)))
        with open("data-{task}/{env_name}/FIM.dat".format(task=task, env_name=env_name), 'rb') as f:
            FIM = pickle.load(f)
            policy.set_FIM(FIM)
        print("Loaded previous checkpoint at step {step}.".format(step=last_checkpoint))
        ewc.consolidate(policy, policy.FIM) # TODO: fix this error: consolidating parameters renames named_params
    else:
        print("Created new policy agent.")

    # Use the Adam optimizer.
    optimizer = torch.optim.Adam(params=policy.parameters(), lr=lr)

    # Run forever.
    episodes = 1100
    try:
        for step in range(episodes):
            # MiniGrid has a QT5 renderer which is pretty cool.
            env.render('human')
            time.sleep(0.01)

            # Run an episode.
            (states, actions, discounted_rewards) = run_episode(env, policy, episode_len)

            # From list to np.array, then save every element in the array
            discounted_rewards_np = np.asarray(discounted_rewards)
            if step % 100 == 0 and training:
                output_reward.write(str(discounted_rewards_np) + "\n")
            avg_reward += np.mean(discounted_rewards)

            if step % 100 == 0:
                print('Average reward @ episode {}: {}'.format(step + int(last_checkpoint), avg_reward / 100))
                if not first_write_flag and training:
                    output_avg.write(str(avg_reward / 100) + "\n")
                else:
                    first_write_flag = False
                avg_reward = 0.0

            # Save the model every 1000 steps
            if step % 500 == 0 and training:
                torch.save(policy.state_dict(), 'torch_models/{env_name}-{task}/model-{env_name}-{step}.pth'.format(
                    env_name=env_name, task=task, step=step + int(last_checkpoint)))
                print("Checkpoint saved.")

            # Repeat each action, and backpropagate discounted
            # rewards. This can probably be batched for efficiency with a
            # memoryless agent...
            if training:
                optimizer.zero_grad()
            for (step, a) in enumerate(actions):
                logits = policy(states[step])
                dist = Categorical(logits=logits)
                loss = -dist.log_prob(actions[step]) * discounted_rewards[step] + ewc.ewc_loss(policy, 2)
                loss.backward()
                if step % 100 == 0 & training:
                    output_loss.write(str(float(loss.data[0])) + "\n")
            if training:
                optimizer.step()
    except KeyboardInterrupt:
        if training and plot:
            utils.plot(task, env_name) # TODO: ensure no file has a blank first line.
        elif training:
            print("Training ended.")
        else:
            print("Simulation ended.")

    # Now estimate the diagonal FIM.
    if need_FIM:
        print('Estimating diagonal FIM...')
        episodes = 1000
        log_probs = []
        for step in range(episodes):
            # Run an episode.
            (states, actions, discounted_rewards) = run_episode(env, policy, episode_len)
            avg_reward += np.mean(discounted_rewards)
            if step % 100 == 0:
                print('Average reward @ episode {}: {}'.format(step, avg_reward / 100))
                avg_reward = 0.0

            # Repeat each action, and backpropagate discounted
            # rewards. This can probably be batched for efficiency with a
            # memoryless agent...
            for (step, a) in enumerate(actions):
                logits = policy(states[step])
                dist = Categorical(logits=logits)
                log_probs.append(-dist.log_prob(actions[step]) * discounted_rewards[step])

        loglikelihoods = torch.cat(log_probs).mean(0)
        loglikelihood_grads = autograd.grad(loglikelihoods, policy.parameters())
        FIM = {n: g**2 for n, g in zip([n for (n, _) in policy.named_parameters()], loglikelihood_grads)}
        for (n, _) in policy.named_parameters():
            FIM[n.replace(".", "__")] = FIM.pop(n)
        with open("data-{task}/{env_name}/FIM.dat".format(task=task, env_name=env_name), 'wb+') as f:
            pickle.dump(FIM, f)
            print("File dumped correctly.")

