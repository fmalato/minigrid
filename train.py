import time
import gym
import numpy as np
import os
import pickle

import torch
from torch.distributions.categorical import Categorical

import ewc
import utils
import network
from network import Policy

def run(episodes=2100, episode_len=50, inner_size=64, lr=0.001, env_name='MiniGrid-Empty-8x8-v0',
        task='task-1',training=False, goal_pos=1, plot=False):

    obs_size = 7*7                        # MiniGrid uses a 7x7 window of visibility.
    act_size = 7                          # Seven possible actions (turn left, right, forward, pickup, drop, etc.)
    avg_reward = 0.0                      # For tracking average regard per episode.
    first_write_flag = True               # Need this due to a weird behavior of the library
    need_diag_FIM = False                 # Avoid the FIM calculus if not required
    need_nondiag_FIM = True               # Same as above but with non diagonal FIM
    model_name = "EWC_model_nondiag_FIM"  # Retrieve the correct model if it exists
    EWC_flag = True                       # If true, uses ewc_loss

    if not EWC_flag:
        need_nondiag_FIM = False
        need_diag_FIM = False
    # Check whether the data directory exists and, if not, create it with all the necessary stuff.
    if not os.path.exists("data-{model}/".format(model=model_name)):
        print("Task 2 data directory created.")
        os.makedirs("data-{model}/".format(model=model_name))

    output_reward = open("data-{model}/reward.txt".format(model=model_name), 'a+')
    output_avg = open("data-{model}/avg_reward.txt".format(model=model_name), 'a+')
    output_loss = open("data-{model}/loss.txt".format(model=model_name), 'a+')

    # Setup OpenAI Gym environment for guessing game.
    env = gym.make(env_name)
    if goal_pos == 2:
        env.set_posX(4)
        env.set_posY(5)

    # Check the model directory
    last_checkpoint = utils.search_last_model('torch_models/', model_name)

    # Instantiate a policy network
    policy = Policy(obs_size=obs_size, act_size=act_size, inner_size=inner_size)

    # If there's a previous checkpoint, load this instead of using a new one.
    if os.listdir('torch_models/{model}/'.format(model=model_name)):
        policy.load_state_dict(torch.load("torch_models/{model}/{model}-{step}.pth".format(
            model=model_name, step=last_checkpoint)))
        if need_diag_FIM and EWC_flag:
            with open("data-{model}/FIM.dat".format(model=model_name), 'rb') as f:
                FIM = pickle.load(f)
                policy.set_FIM(FIM)
        elif need_nondiag_FIM and EWC_flag:
            with open("data-{model}/nonD_FIM.dat".format(model=model_name), 'rb') as f:
                FIM = pickle.load(f)
                policy.set_FIM(FIM)
        print("Loaded previous checkpoint at step {step}.".format(step=last_checkpoint))

    else:
        print("Created new policy agent.")

    # Use the Adam optimizer.
    optimizer = torch.optim.Adam(params=policy.parameters(), lr=lr)

    try:
        for step in range(episodes):
            # MiniGrid has a QT5 renderer which is pretty cool.
            env.render('human')
            time.sleep(0.01)

            # Run an episode.
            (states, actions, discounted_rewards) = network.run_episode(env, policy, episode_len)

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
                torch.save(policy.state_dict(), 'torch_models/{model}/{model}-{step}.pth'.format(
                    model=model_name, step=step + int(last_checkpoint)))
                print("Checkpoint saved.")

            # Repeat each action, and backpropagate discounted
            # rewards. This can probably be batched for efficiency with a
            # memoryless agent...
            if training:
                optimizer.zero_grad()
            episode_loss = []
            for (step, a) in enumerate(actions):
                logits = policy(states[step])
                dist = Categorical(logits=logits)
                if EWC_flag:
                    loss = -dist.log_prob(actions[step]) * discounted_rewards[step] + ewc.ewc_loss(policy, 2)
                else:
                    loss = -dist.log_prob(actions[step]) * discounted_rewards[step]
                loss.backward()
                episode_loss.append(loss.data[0])
            current_loss = sum([x for x in episode_loss]) / episode_len
            if training:
                optimizer.step()
                output_loss.write(str(float(current_loss)) + "\n")

    except KeyboardInterrupt:
        if training:
            print("Training ended.")
        else:
            print("Simulation ended.")

    # Now estimate the diagonal FIM.
    if need_diag_FIM:
        utils.diagonal_FIM(policy, env, episode_len, model_name)
    elif need_nondiag_FIM:
        utils.non_diagonal_FIM(policy, env, episode_len, model_name)

run(episodes=2100, episode_len=50, task='task-2', env_name='MiniGrid-Empty-8x8-v0', goal_pos=1, training = True)

