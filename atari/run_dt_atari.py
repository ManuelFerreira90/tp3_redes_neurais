import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
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
from create_dataset import create_dataset
from datasets import load_dataset
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=50)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

set_seed(args.seed)

# class StateActionReturnDataset(Dataset):

#     def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
#         self.block_size = block_size
#         self.vocab_size = int(max(actions)) + 1
#         self.data = data
#         self.actions = actions
#         self.done_idxs = done_idxs
#         self.rtgs = rtgs
#         self.timesteps = timesteps
    
#     def __len__(self):
#         return len(self.data) - self.block_size

#     def __getitem__(self, idx):
#         block_size = self.block_size // 3
#         done_idx = idx + block_size
#         for i in self.done_idxs:
#             if i > idx: # first done_idx greater than idx
#                 done_idx = min(int(i), done_idx)
#                 break
#         idx = done_idx - block_size
#         states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
#         states = states / 255.
#         actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
#         rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
#         timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

#         return states, actions, rtgs, timesteps
class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = int(max(actions)) + 1
        
        self.data = data 
        self.actions = actions
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3 # = 30
        
        states = torch.from_numpy(self.data[idx : idx + block_size]).to(dtype=torch.float32)
        states = states / 255.
        
        actions = torch.tensor(self.actions[idx : idx + block_size], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.rtgs[idx : idx + block_size], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx : idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps


local_dataset_path = 'dataset_breakout.npy'
print(f"Loading data from local file: {local_dataset_path}...")


data_dict = np.load(local_dataset_path, allow_pickle=True).item()

obss = data_dict['observations'].squeeze()
actions = data_dict['actions'].squeeze()
stepwise_returns = data_dict['rewards'].squeeze()
dones = data_dict['dones'].squeeze()

done_idxs = np.where(dones)[0]

rtgs = np.zeros_like(stepwise_returns, dtype=float)
start_index = 0
for i in done_idxs:
    i = int(i)
    traj_returns = stepwise_returns[start_index:i+1]
    for j in range(len(traj_returns) - 1, -1, -1):
        rtgs[start_index + j] = traj_returns[j:].sum()
    start_index = i + 1

timesteps = np.zeros(len(actions), dtype=int)
start_index = 0
for i in done_idxs:
    i = int(i)
    timesteps[start_index:i+1] = np.arange(i + 1 - start_index)
    start_index = i + 1

print("Data loaded and processed successfully!")


# max_test_steps = 10000  # Usaremos apenas os primeiros 10.000 passos

# local_dataset_path = 'dataset_pong.npy'
# print(f"Loading data from local file: {local_dataset_path}...")

# data_dict = np.load(local_dataset_path, allow_pickle=True).item()

# # Corta os dados para o tamanho de teste
# obss = data_dict['observations'][:max_test_steps].squeeze()
# actions = data_dict['actions'][:max_test_steps].squeeze()
# stepwise_returns = data_dict['rewards'][:max_test_steps].squeeze()
# dones = data_dict['dones'][:max_test_steps].squeeze()

# # O resto do processamento continua igual, mas agora com menos dados
# done_idxs = np.where(dones)[0]
# rtgs = np.zeros_like(stepwise_returns, dtype=float)
# start_index = 0
# for i in done_idxs:
#     i = int(i)
#     traj_returns = stepwise_returns[start_index:i+1]
#     for j in range(len(traj_returns) - 1, -1, -1):
#         rtgs[start_index + j] = traj_returns[j:].sum()
#     start_index = i + 1

# timesteps = np.zeros(len(actions), dtype=int)
# start_index = 0
# for i in done_idxs:
#     i = int(i)
#     timesteps[start_index:i+1] = np.arange(i + 1 - start_index)
#     start_index = i + 1

# returns = np.zeros_like(done_idxs)

# print(f"Data loaded for test! Using {len(actions)} steps.")


# Configuração do logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)


train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, rtgs, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=int(max(timesteps)))

# teste
# mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
#                   n_layer=2, n_head=4, n_embd=64, model_type=args.model_type, max_timestep=int(max(timesteps)))

model = GPT(mconf)

# Configuração do treinador e início do treinamento
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=0, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=int(max(timesteps)))
trainer = Trainer(model, train_dataset, None, tconf)

print("\n--- Starting Training ---")
trainer.train()
print("\n--- Training Finished ---")

