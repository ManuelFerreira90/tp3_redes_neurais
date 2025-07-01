import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from fixed_replay_buffer import FixedReplayBuffer

# def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
#     # -- load data from memory (make more efficient)
#     obss = []
#     actions = []
#     returns = [0]
#     done_idxs = []
#     stepwise_returns = []

#     transitions_per_buffer = np.zeros(50, dtype=int)
#     num_trajectories = 0
#     while len(obss) < num_steps:
#         buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
#         i = transitions_per_buffer[buffer_num]
#         print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
#         frb = FixedReplayBuffer(
#             data_dir=data_dir_prefix + game + '/1/replay_logs',
#             replay_suffix=buffer_num,
#             observation_shape=(84, 84),
#             stack_size=4,
#             update_horizon=1,
#             gamma=0.99,
#             observation_dtype=np.uint8,
#             batch_size=32,
#             replay_capacity=100000)
#         if frb._loaded_buffers:
#             done = False
#             curr_num_transitions = len(obss)
#             trajectories_to_load = trajectories_per_buffer
#             while not done:
#                 states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
#                 states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
#                 obss += [states]
#                 actions += [ac[0]]
#                 stepwise_returns += [ret[0]]
#                 if terminal[0]:
#                     done_idxs += [len(obss)]
#                     returns += [0]
#                     if trajectories_to_load == 0:
#                         done = True
#                     else:
#                         trajectories_to_load -= 1
#                 returns[-1] += ret[0]
#                 i += 1
#                 if i >= 100000:
#                     obss = obss[:curr_num_transitions]
#                     actions = actions[:curr_num_transitions]
#                     stepwise_returns = stepwise_returns[:curr_num_transitions]
#                     returns[-1] = 0
#                     i = transitions_per_buffer[buffer_num]
#                     done = True
#             num_trajectories += (trajectories_per_buffer - trajectories_to_load)
#             transitions_per_buffer[buffer_num] = i
#         print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

#     actions = np.array(actions)
#     returns = np.array(returns)
#     stepwise_returns = np.array(stepwise_returns)
#     done_idxs = np.array(done_idxs)

#     # -- create reward-to-go dataset
#     start_index = 0
#     rtg = np.zeros_like(stepwise_returns)
#     for i in done_idxs:
#         i = int(i)
#         curr_traj_returns = stepwise_returns[start_index:i]
#         for j in range(i-1, start_index-1, -1): # start from i-1
#             rtg_j = curr_traj_returns[j-start_index:i-start_index]
#             rtg[j] = sum(rtg_j)
#         start_index = i
#     print('max rtg is %d' % max(rtg))

#     # -- create timestep dataset
#     start_index = 0
#     timesteps = np.zeros(len(actions)+1, dtype=int)
#     for i in done_idxs:
#         i = int(i)
#         timesteps[start_index:i+1] = np.arange(i+1 - start_index)
#         start_index = i+1
#     print('max timestep is %d' % max(timesteps))

#     return obss, actions, returns, done_idxs, rtg, timesteps

def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    
    all_obs = []
    all_actions = []
    all_rewards = []
    all_episode_starts = []
    returns = []

    base_filename = f'{game}NoFrameskip-v4'
    
    for buffer_num in range(11):
        try:
            file_path = os.path.join(data_dir_prefix, game, f'{base_filename}_{buffer_num}.npz')
            print(f"Loading from: {file_path}")
            data = np.load(file_path)

            all_obs.extend(data['obs'])
            all_actions.extend(data['taken actions'])
            all_rewards.extend(data['rewards'])
            all_episode_starts.extend(data['episode_starts'])
            
            print(f"  Buffer {buffer_num} loaded successfully.")

        except Exception as e:
            print(f"  Error loading buffer {buffer_num}: {e}. Skipping.")


    stepwise_returns = np.array(all_rewards)
    actions = np.array(all_actions)

    done_idxs = []
    for i in range(len(all_episode_starts) - 1):
        if all_episode_starts[i+1]:
            done_idxs.append(i)
    
    done_idxs.append(len(actions) - 1)
    
    start_index = 0
    for i in done_idxs:
        i = int(i)
        returns.append(stepwise_returns[start_index:i+1].sum())
        start_index = i + 1

    done_idxs = np.array(done_idxs)
    returns = np.array(returns)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns, dtype=float)
    for i in done_idxs:
        i = int(i)
        traj_returns = stepwise_returns[start_index:i+1]
        for j in range(len(traj_returns) - 1, -1, -1):
            rtg[start_index + j] = traj_returns[j:].sum()
        start_index = i + 1
    print('max rtg is %f' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions), dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i + 1 - start_index)
        start_index = i + 1
    print('max timestep is %d' % max(timesteps))

    return all_obs, actions, returns, done_idxs, rtg, timesteps
