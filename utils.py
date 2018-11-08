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
def search_last_model(path, model_name):

    checkpoints = []
    try:
        if not os.path.exists("{path}{model}/".format(path=path, model=model_name)):
            print("Directory for {model} models not found: a new one has been created.".format(model=model_name))
            os.makedirs("{path}{model}/".format(path=path, model=model_name))
            return 0

        # Execute this line only if there's a directory
        file_list = os.listdir("{path}{model}/".format(path=path, model=model_name))

        for fname in file_list:
            res = re.findall("{model}-(\d+).pth".format(model=model_name), fname)
            checkpoints.append(res[0])

        checkpoints_np = np.asarray(checkpoints, dtype=int)
        max_index = checkpoints_np.argmax()
        max_arg = checkpoints_np[max_index]

        return max_arg
    except ValueError:
        print("Directory for {model} models not found: a new one has been created.".format(model=model_name))
        os.makedirs("{path}{model}/".format(path=path, model=model_name))
        return 0

# Provide a way to know how many lines a file has.
def file_len(fname):

    with open(fname) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1

# Plot all the important stuff like loss, episode reward and average reward
def plot(model_name):

    avg_len = file_len("data-{model}/avg_reward.txt".format(model=model_name))
    rew_len = file_len("data-{model}/reward.txt".format(model=model_name))
    loss_len = file_len("data-{model}/loss.txt".format(model=model_name))
    avg_x = []
    rew_x = []
    loss_x = []
    avg_d = []
    rew_d = []
    loss_d = []
    checkpoint = search_last_model("torch_models/", model_name)

    for y in range(avg_len):
        avg_x.append(y * 100)
    for y in range(rew_len):
        rew_x.append(y)
    for y in range(loss_len):
        loss_x.append(y)
    with open("data-{model}/avg_reward.txt".format(model=model_name)) as file:
        data_avg_reward = file.readlines()
    rewards = csv.reader(data_avg_reward)
    with open("data-{model}/loss.txt".format(model=model_name)) as f_loss:
        data_loss = f_loss.readlines()
    loss = csv.reader(data_loss)
    with open("data-{model}/reward.txt".format(model=model_name)) as f_rew:
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

def non_diagonal_FIM(agent, env, episode_len, model_name):
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
    loglikelihood_grads = []
    loglikelihoods = torch.cat(log_probs).mean(0)
    for (n, p) in agent.named_parameters():
        loglikelihood_grads.append(autograd.grad(loglikelihoods, p, retain_graph=True))
    # A very inelegant way to delete biases
    loglikelihood_grads.pop(3)
    loglikelihood_grads.pop(1)
    labels = ['layer_1', 'layer_2']
    FIM = {label: x for label, x in zip(labels, [torch.mm(x[0], x[0].t()) for x in loglikelihood_grads])}

    with open("data-{model}/nonD_FIM.dat".format(model=model_name), 'wb+') as f:
        pickle.dump(FIM, f)
        print("File dumped correctly.")

def diagonal_FIM(agent, env, episode_len, model_name):
    print('Estimating diagonal FIM...')
    episodes = 1000
    log_probs = []
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
    # torch.dot(loglikelihood_grads * loglikelihood_grads.T)
    FIM = {n: g**2 for n, g in zip([n for (n, _) in agent.named_parameters()], loglikelihood_grads)}
    for (n, _) in agent.named_parameters():
        FIM[n.replace(".", "__")] = FIM.pop(n)
    with open("data-{model}/FIM.dat".format(model=model_name), 'wb+') as f:
        pickle.dump(FIM, f)
        print("File dumped correctly.")





