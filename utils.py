import os
import re
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle

def search_last_model(path, env_name, task):

    checkpoints = []

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

def file_len(fname):

    with open(fname) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1

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







