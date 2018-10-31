import time
import gym
import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from itertools import count
import utils
import main

def run(episode_len=50, inner_size=64, lr=0.001, env_name='MiniGrid-Empty-8x8-v0', task='task-1', training=False,
        goal_pos=1, plot=False):

    episode_len = episode_len            # Length of each game.
    obs_size = 7 * 7                     # MiniGrid uses a 7x7 window of visibility.
    act_size = 7                         # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
    inner_size = inner_size              # Number of neurons in two hidden layers.
    lr = lr                              # Adam learning rate
    avg_reward = 0.0                     # For tracking average regard per episode.
    env_name = env_name                  # Size of the grid
    task = task                          # Task name for saving data-task-1 on the right folder
    first_write_flag = True              # Need this due to a weird behavior of the library
    training = training                  # If set to False, optimizer won't run (and the net won't learn)
    goal_pos = goal_pos                  # Bottom right if 1, bottom left if 2
    plot = plot                          # Plots the results

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
        env.set_posX(4)
        env.set_posY(5)

    # Check the model directory
    last_checkpoint = utils.search_last_model('torch_models/', env_name, task)

    # Instantiate a policy network
    policy = main.Policy(obs_size=obs_size, act_size=act_size, inner_size=inner_size)

    # If there's a previous checkpoint, load this instead of using a new one.
    if os.listdir('torch_models/{env}-{task}/'.format(env=env_name, task=task)):
        policy.load_state_dict(torch.load("torch_models/{env_name}-{task}/model-{env_name}-{step}.pth".format(
            env_name=env_name, step=last_checkpoint, task=task)))
        print("Loaded previous checkpoint at step {step}.".format(step=last_checkpoint))
    else:
        print("Created new policy agent.")

    # Use the Adam optimizer.
    optimizer = torch.optim.Adam(params=policy.parameters(), lr=lr)

    # Run forever.
    try:
        for step in count():
            # MiniGrid has a QT5 renderer which is pretty cool.
            env.render('human')
            time.sleep(0.01)

            # Run an episode.
            (states, actions, discounted_rewards) = main.run_episode(env, policy, episode_len)

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
                loss = -dist.log_prob(actions[step]) * discounted_rewards[step]
                loss.backward()
                if step % 100 == 0 & training:
                    output_loss.write(str(float(loss.data[0])) + "\n")  # TODO: check loss values. Seriously.
            if training:
                optimizer.step()
    except KeyboardInterrupt:
        if training and plot:
            utils.plot(task, env_name)  # TODO: ensure no file has a blank first line. If one has, it's error.
        elif training:
            print("Training ended.")
        else:
            print("Simulation ended.")

run(episode_len=50, task='task-2', env_name='MiniGrid-Empty-8x8-v0', goal_pos=1, training = True)

