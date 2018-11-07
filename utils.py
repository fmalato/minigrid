import os
import re
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle

import torch
import torch.autograd as autograd
from torch.distributions.categorical import Categorical

import main

# Search for the last checkpoint inside the current environment's models directory by scanning the
# number at the end of the file name. Store all the numbers in an array and return the argmax.
def search_last_model(path, env_name, task):

    checkpoints = []
    try:
        if not os.path.exists("{path}{env_name}-{task}/".format(path=path, env_name=env_name, task=task)):
            print("Directory for {env}-{task} models not found: a new one has been created.".format(env=env_name,
                                                                                                    task=task))
            os.makedirs("{path}{env_name}-{task}/".format(path=path, env_name=env_name, task=task))
            return 0

        # Execute this line only if there's a directory
        file_list = os.listdir("{path}{env_name}-{task}/".format(path=path, env_name=env_name, task=task))

        for fname in file_list:
            res = re.findall("{env}-(\d+).pth".format(env=env_name), fname)
            checkpoints.append(res[0])

        checkpoints_np = np.asarray(checkpoints, dtype=int)
        max_index = checkpoints_np.argmax()
        max_arg = checkpoints_np[max_index]

        return max_arg
    except ValueError:
        print("Directory for {env}-{task} models not found: a new one has been created.".format(env=env_name,
                                                                                                task=task))
        os.makedirs("{path}{env_name}-{task}/".format(path=path, env_name=env_name, task=task))
        return 0

# Provide a way to know how many lines a file has.
def file_len(fname):

    with open(fname) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1

# Plot all the important stuff like loss, episode reward and average reward
def plot(task, env_name):

    avg_len = file_len("data-{task}/{env_name}/avg_reward.txt".format(task=task, env_name=env_name))
    rew_len = file_len("data-{task}/{env_name}/reward.txt".format(task=task, env_name=env_name))
    loss_len = file_len("data-{task}/{env_name}/loss.txt".format(task=task, env_name=env_name))
    avg_x = []
    rew_x = []
    loss_x = []
    avg_d = []
    rew_d = []
    loss_d = []
    checkpoint = search_last_model("torch_models/", env_name, task)

    for y in range(avg_len):
        avg_x.append(y * 100)
    for y in range(rew_len):
        rew_x.append(y)
    for y in range(loss_len):
        loss_x.append(y)
    with open("data-{task}/{env_name}/avg_reward.txt".format(task=task, env_name=env_name)) as file:
        data_avg_reward = file.readlines()
    rewards = csv.reader(data_avg_reward)
    with open("data-{task}/{env_name}/loss.txt".format(task=task, env_name=env_name)) as f_loss:
        data_loss = f_loss.readlines()
    loss = csv.reader(data_loss)
    with open("data-{task}/{env_name}/reward.txt".format(task=task, env_name=env_name)) as f_rew:
        data_rew = f_rew.readlines()
    rew = csv.reader(data_rew)

    for n in list(rewards):
        avg_d.append(float(n[0]))
    for n in list(loss):
        loss_d.append(float(n[0]))
    """for n in list(rew):
        rew_d.append(float(n[0]))"""    # TODO: data are saved in a strange format, need to check it before plotting

    plt.plot(avg_x, avg_d)
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.show()
    # plt.plot(rew_x, rew_d)
    plt.plot(loss_x, loss_d)
    plt.xlabel("Episode")
    plt.ylabel("Loss per episode")
    plt.show()

def non_diagonal_FIM(agent, env, episode_len, env_name, task):
    print('Estimating non-diagonal FIM...')
    episodes = 1000
    log_probs = []
    avg_reward = 0.0
    for step in range(episodes):
        # Run an episode.
        (states, actions, discounted_rewards) = main.run_episode(env, agent, episode_len)
        avg_reward += np.mean(discounted_rewards)
        if step % 100 == 0:
            print('Average reward @ episode {}: {}'.format(step, avg_reward / 100))
            avg_reward = 0.0

        # Repeat each action, and backpropagate discounted
        # rewards. This can probably be batched for efficiency with a
        # memoryless agent...
        for (step, a) in enumerate(actions):
            logits = agent(states[step])
            dist = Categorical(logits=logits)
            log_probs.append(-dist.log_prob(actions[step]) * discounted_rewards[step])

    loglikelihoods = torch.cat(log_probs).mean(0)
    loglikelihood_grads = autograd.grad(loglikelihoods, agent.parameters())
    loglike_list = [x for x in loglikelihood_grads]
    # TODO: understand what's inside loglikelihood and why there are tensors with different size
    FIM = [torch.mm(x, x.t()) for x in loglike_list]
    with open("data-{task}/{env_name}/nonD_FIM.dat".format(task=task, env_name=env_name), 'wb+') as f:
        pickle.dump(FIM, f)
        print("File dumped correctly.")





