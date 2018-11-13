import os
import torch
import numpy as np
import pickle
import time
import gym
from torch.distributions.categorical import Categorical

import utils
import ewc
from network import Policy
from network import run_episode

def test(model_name, goal_pos=1, EWC_flag=True):

    episode_len = 50                      # Length of each game.
    obs_size = 7*7                        # MiniGrid uses a 7x7 window of visibility.
    act_size = 7                          # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
    inner_size = 64                       # Number of neurons in two hidden layers.
    avg_reward = 0.0                      # For tracking average regard per episode.
    env_name = 'MiniGrid-Empty-8x8-v0'    # Size of the grid

    #test_avg_reward = open("data-{model}/test_avg_rewards.txt".format(model=model_name), 'a+')

    # Setup OpenAI Gym environment for guessing game.
    env = gym.make(env_name)
    if goal_pos == 2:
        env.set_posX(4)
        env.set_posY(5)

    # Check the model directory
    last_checkpoint = utils.search_last_model('torch_models/', model_name)

    # Instantiate a policy network
    policy = Policy(obs_size=obs_size, act_size=act_size, inner_size=inner_size)

    policy.load_state_dict(torch.load("torch_models/{model}/{model}-{step}.pth".format(
        model=model_name, step=last_checkpoint)))
    if EWC_flag:
        try:
            with open("data-{model}/FIM.dat".format(model=model_name), 'rb') as f:
                FIM = pickle.load(f)
            policy.set_FIM(FIM)
        except FileNotFoundError:
            with open("data-{model}/nonD_FIM.dat".format(model=model_name), 'rb') as f:
                FIM = pickle.load(f)
            policy.set_FIM(FIM)
    print("Loaded previous checkpoint at step {step}.".format(step=last_checkpoint))

    # Run forever.
    episodes = 1001
    for step in range(episodes):
        # MiniGrid has a QT5 renderer which is pretty cool.
        env.render('human')
        time.sleep(0.01)

        # Run an episode.
        (states, actions, discounted_rewards) = run_episode(env, policy, episode_len)
        avg_reward += np.mean(discounted_rewards)

        if step % 100 == 0:
            print('Average reward @ episode {}: {}'.format(step + int(last_checkpoint), avg_reward / 100))
            #if step != 0:
                #test_avg_reward.write(str(avg_reward / 100) + "\n")
            avg_reward = 0.0

def complete_test_cycle(model_name, EWC_flag):

    print("Testing {model} on goal_pos 1.".format(model=model_name))
    test(model_name, 1, EWC_flag)
    time.sleep(1)
    print("Testing {model} on goal_pos 2.".format(model=model_name))
    test(model_name, 2, EWC_flag)
    time.sleep(1)
    print("Test complete.")

complete_test_cycle("EWC_model_diag_FIM", True)
complete_test_cycle("EWC_model_nondiag_FIM", True)
complete_test_cycle("non_EWC_model", False)
