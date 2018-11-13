import os
import re
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
from PIL import Image

import torch
import torch.autograd as autograd
from torch.distributions.categorical import Categorical

import network


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


def non_diagonal_FIM(agent, env, episode_len, model_name):
    print('Estimating non-diagonal FIM...')
    episodes = 1000
    log_probs = []
    avg_reward = 0.0
    for step in range(episodes):
        # Run an episode.
        (states, actions, discounted_rewards) = network.run_episode(env, agent, episode_len)
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
        (states, actions, discounted_rewards) = network.run_episode(env, agent, episode_len)
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


def test_plot():

    x_axis = np.arange(0, 4000, 100)
    y_axis = []
    models = ['EWC_model_diag_FIM', 'EWC_model_nondiag_FIM', 'non_EWC_model']
    y_data = []

    for model_name in models:
        with open("data-{model}/avg_reward.txt".format(model=model_name)) as file:
            test_data = file.readlines()
        t_d = csv.reader(test_data)
        for n in list(t_d):
            y_axis.append(float(n[0]))
        with open("data-{model}/test_avg_rewards.txt".format(model=model_name)) as file:
            test_data = file.readlines()
        t_d = csv.reader(test_data)
        for n in list(t_d):
            y_axis.append(float(n[0]))
        y_data.append(y_axis)
        y_axis = []
    for el in y_data:
        plt.ylabel("Ricompensa media")
        plt.xlabel("Episodio")
        plt.plot(x_axis, el)
    plt.title('Comparazione dei modelli')
    plt.legend(('EWC con FIM diagonale', 'EWC con FIM non diagonale', 'senza EWC'))
    # plt.savefig('images/training_comparison.png')
    plt.show()


def loss_plot():

    x_axis = np.arange(0, 4200, 1)
    y_axis = []
    models = ['EWC_model_diag_FIM', 'EWC_model_nondiag_FIM', 'non_EWC_model']
    y_data = []

    for model_name in models:
        with open("data-{model}/loss.txt".format(model=model_name)) as file:
            test_data = file.readlines()
        t_d = csv.reader(test_data)
        for n in list(t_d):
            y_axis.append(float(n[0]))
        y_data.append(y_axis)
        y_axis = []
    for el in y_data:
        plt.ylabel("Loss")
        plt.xlabel("Episodio")
        plt.plot(x_axis, el)
    plt.title('Comparazione dei modelli')
    plt.legend(('EWC con FIM diagonale', 'EWC con FIM non diagonale', 'senza EWC'))
    # plt.savefig('images/loss_comparison.png')
    plt.show()


def FIM_to_image(filepath):

    with open(filepath, 'rb') as f:
        FIM = pickle.load(f)
        labels = ['affine1_weight', 'affine2_weight']
        correct_FIM = {label: x for label, x in zip(labels, [FIM[y] for y in labels])} # TODO: what's this?!
    FIM_list = [[]]
    col = 0
    for key in FIM.keys():
        for item in FIM[key]:
            for el in item.data:
                x = float(el)
                FIM_list[col].append(x)
        col += 1
    array = np.fromiter(FIM_list, dtype=float)
    img = Image.fromarray(array, 'RGB')
    img.save('my.png')
    img.show()

FIM_to_image('data-EWC_model_diag_FIM/FIM.dat')



