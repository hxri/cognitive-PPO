import os
import argparse
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from appraisal import motivational_relevance, novelty, certainity
import matplotlib.pyplot as plt
from collections import deque
from emotion import stress, emo_app, norm_app, stress
from collections import Counter
import cv2


# Environment parameters
agent_view_size = 7
max_steps = 100
n_obstacles = 7
size = 10
agent_start_pos = None  # Dynamic start position

# Make vectorized environment function
def make_env(gym_id, seed):
    def thunk():
        # env = gym.make(gym_id)
        env = gym.make(gym_id,
                       render_mode="human",
                       max_steps=max_steps,
                       agent_view_size=agent_view_size, 
                       n_obstacles=n_obstacles,
                       size=size,
                       agent_start_pos=agent_start_pos,
                       dynamic_wall=False,
                       dynamic_goal=True,
                       dynamic_obstacles=True,
                       moving_goal=True,
                       n_goals=1,
                       wall_split=2,
                       agent_pov=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed) 
        return env
    return thunk

def minmax(score):
    return (score + 1) / 2

def count_sequence_occurrences(numbers, sequence):
    count = 0
    sequence_length = len(sequence)
    for i in range(len(numbers) - sequence_length + 1):
        if numbers[i:i + sequence_length] == sequence:
            count += 1
    return count

def count_string_occurrences(strings):
    occurrences = {}
    for string in strings:
        if string in occurrences:
            occurrences[string] += 1
        else:
            occurrences[string] = 1
    return occurrences

# Weight and Bias intialization for PPO
def layer_inint(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def appraisal_calc(obs=None, logits=None):
    a1 = motivational_relevance(obs)
    a2 = novelty(logits)
    a3 = certainity(logits)
    app = torch.stack((a1, a2, a3), -1)
    return app

def reward_with_app(base_rw, mot_rel, nov):
    new_rew = torch.where(base_rw == -1, base_rw, (base_rw + mot_rel + nov)/3)
    return new_rew

class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(query.size(0), -1, query.size(2) * query.size(3)).permute(0, 2, 1)
        key = key.view(key.size(0), -1, key.size(2) * key.size(3)).permute(0, 2, 1)
        value = value.view(value.size(0), -1, value.size(2) * value.size(3))

        attention = F.softmax(torch.bmm(query, key.transpose(1, 2)), dim=2)
        out = torch.bmm(attention, value.transpose(1, 2))

        out = out.view(*x.size())
        out = self.gamma * out + x
        return out

# Agent model
class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()

        self.appraisal_size = 6

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            Attention(64),
            nn.ReLU(),
            nn.Flatten()
        )

        input_shape = (agent_view_size + 3, agent_view_size + 3)
        self.critic = nn.Sequential(
            nn.Linear(64*input_shape[0]*input_shape[1] + 6, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(64*input_shape[0]*input_shape[1] + 0, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, envs.single_action_space.n)
        )
        self.NRE = nn.Sequential(
            nn.Linear(64*input_shape[0]*input_shape[1] + envs.single_action_space.n, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    # def get_appraisal(self, x, temp, sts):
    #     x1 = self.conv(x.permute(0, 3, 1, 2))
    #     if(args.monitor_only):
    #         app = appraisal_calc(x[:][..., 0], self.actor(x1))
    #     else:
    #         # temp = torch.cat([temp, sts], dim=-1)
    #         # x2 = torch.mean((xe.unsqueeze(2) * temp.unsqueeze(1)), dim=2)
    #         # xe = (x2 - torch.min(x2)) / (torch.max(x2) - torch.min(x2))
    #         xe = torch.cat([x1, temp], dim=-1)
    #         app = appraisal_calc(x[:][..., 0], self.actor(x1))
    #     return app

    def get_value(self, x, appraisal, sts):
        x1 = self.conv(x.permute(0, 3, 1, 2))
        if(args.monitor_only):
            xe = x1
        else:
            # appraisal = torch.cat([appraisal, sts], dim=-1)
            # x2 = torch.mean((x1.unsqueeze(2) * appraisal.unsqueeze(1)), dim=2)
            # xe = (x2 - torch.min(x2)) / (torch.max(x2) - torch.min(x2))
            xe = torch.cat([x1, appraisal], dim=-1)
        cr = self.actor(x1)
        return self.critic(xe)
            
    def get_action_and_value(self, x, appraisal, sts, action=None):
        x1 = self.conv(x.permute(0, 3, 1, 2))
        # print(torch.min(x1))
        if(args.monitor_only):
            xe = x1
        else:
            # appraisal = torch.cat([appraisal, sts], dim=-1)
            # x2 = torch.mean((x1.unsqueeze(2) * appraisal.unsqueeze(1)), dim=2)
            # xe = (x2 - torch.min(x2)) / (torch.max(x2) - torch.min(x2))
            xe = torch.cat([x1, appraisal], dim=-1)
        logits = self.actor(x1)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        n_reward = self.NRE(torch.cat([x1, logits], dim=-1)).squeeze(1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(xe), n_reward, logits


# Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', 
                        type=str, 
                        default=os.path.basename(__file__).rstrip('.py'), 
                        help='Name of Experiment')
    parser.add_argument('--gym-id',
                        type=str,
                        default='MiniGrid-Dynamic-Obstacles-Random-10x10-v0', # MiniGrid-Empty-5x5-v0, MiniGrid-Dynamic-Obstacles-8x8-v0
                        help='ID of Gym Environment')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Seed for the experiment')
    parser.add_argument('--total-timesteps',
                        type=int,
                        default=50000,
                        help='Total number of timesteps')
    parser.add_argument('--torch-det',
                        type=lambda x:bool(strtobool(x)),
                        default=True,
                        nargs='?',
                        const=True,
                        help='If toggled, `torch.backend.cudnn.determentistic=False`')
    parser.add_argument('--cuda',
                        type=lambda x:bool(strtobool(x)),
                        default=True,
                        nargs='?',
                        const=True,
                        help='If toggled, cuda will not be used')
    parser.add_argument('--monitor-only',
                        type=lambda x:bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='If toggled, enables appraisal monitoring only mode')
    
    # Algorithm specific
    parser.add_argument('--num-envs',
                        type=int,
                        default=1,
                        help='Number of parallel gym envs')
    parser.add_argument('--num-steps',
                        type=int,
                        default=128,
                        help='Number of steps to run each environment per policy rollout')
    parser.add_argument('--num-minibatches',
                        type=int,
                        default=4,
                        help='Number of mini-batches')
    
    

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_det

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only descrete action space is supported"

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(f'runs/{args.gym_id}__{args.exp_name}/trained_agent.pt'))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Algorithm specific setup

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space['image'].shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    appraisals = torch.zeros((args.num_steps, args.num_envs) + (6,)).to(device)
    n_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

    gc = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cp = torch.zeros((args.num_steps, args.num_envs)).to(device)
    anti = torch.zeros((args.num_steps, args.num_envs)).to(device)
    sts = torch.zeros((args.num_steps, args.num_envs) + (1,)).to(device)

    # Try not to modify
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]['image']).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    temp_app = torch.cat((torch.tensor([[0.1]]),
                          torch.tensor([[0.1]]),
                          torch.tensor([[0.1]]),
                          torch.tensor([[0.1]]),
                          torch.tensor([[0.1]]),
                          torch.tensor([[0.1]]),), dim=1)
    next_sts = torch.tensor([[0.1]])
    # next_appraisal = torch.tensor(agent.get_appraisal(next_obs, temp_app, next_sts))
    next_appraisal = temp_app # torch.cat((next_appraisal, torch.tensor([[0.1]]), torch.tensor([[0.1]]), torch.tensor([[0.1]])), dim=1)

    return_arr = []
    gc_prev = 0
    cp_prev = 0
    flag = 0 

    emotion_arr = []
    sts_arr = []
    aversions = []
    act_arr = []
    action_arr = []
    agent_pos_arr = []
    app_data = []

    for update in range(1, num_updates + 1):
        start_time = time.time()
        stress_level = []
        for step in range(0, args.num_steps):
            flag = 0 
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # Action logic
            with torch.no_grad():
                action, logprob, _, value, n_reward, logits = agent.get_action_and_value(next_obs, next_appraisal, next_sts)
                # print(action)
                values[step] = value.flatten()
            actions[step] = action
            act_arr.append(action.detach().numpy()[0])
            action_arr.append(action.detach().numpy()[0])
            logprobs[step] = logprob
            n_rewards[step] = n_reward
            app = appraisal_calc(next_obs[:][..., 0], logits)
            # appraisals[step] = app
            # mot_rel = torch.stack([item[0] for item in app])
            # nov = torch.stack([item[1] for item in app])normminmax_val

            # Ty not to modify: execute game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            base_rw = torch.tensor(reward).to(device).view(-1)

            if('agent_pos' in info):
                agent_pos_arr.append(info['agent_pos'][0])

            if('final_info' in info):
                if('goal_congruence' in info['final_info'][0]):
                    gc[step] = torch.tensor(info['final_info'][0]['goal_congruence'])
                    cp[step] = torch.tensor(info['final_info'][0]['coping_potential'])
                    gc_prev = gc[step]
                    cp_prev = cp[step]
                else:
                    gc[step] = gc_prev
                    cp[step] = cp_prev
                    flag = 1
            else:
                gc[step] = torch.tensor(info['goal_congruence'])
                cp[step] = torch.tensor(info['coping_potential'])

            
            # print(np.average(reward))
            rw = torch.tensor(reward).to(device).view(-1)
            rewards[step] = rw - (0.01 * (1-app[:, 0])) - (0.1 * (1-gc[step])) # - 0.1 * (1-app[:, 0]) #- 0.01 * ((1-gc[step]) + (1-app[:, 0])) # (0.1 * (1-app[:, 0]) + 0.1 * (cp[step])) # - 0.1 * (1-cp[step]) - 0.1 * (app[0][1])
            if(step == 0):
                anti[step] = 1 - torch.abs(rewards[step] -  0)
            else:
                anti[step] = 1 - torch.abs(rewards[step] -  n_rewards[step-1])
            app = torch.cat((app, gc[step].unsqueeze(1), cp[step].unsqueeze(1), anti[step].unsqueeze(1)), dim=1)
            app_data.append(app.detach().numpy()[0])
            stress_level.append(stress(app))
            sts_arr.append(stress(app))
            sts[step] = stress(app)
            next_sts = sts[step]
            appraisals[step] = app
            next_appraisal = app

            # print(emo_app(app[0]), app)
                  
            emotion_arr.append(emo_app(app[0]))
        
            next_obs, next_done = torch.Tensor(next_obs['image']).to(device), torch.Tensor(terminated).to(device)
            time.sleep(.1)
            if('final_info' in info):
                end_time = time.time()
                for item in info['final_info']:
                    if(item):
                        if 'episode' in item.keys():
                            return_arr.append(item['episode']['r'])
                            print(f"global_step={global_step}, episodic_return={item['episode']['r']}, stress_level={np.average(stress_level)}")
                            aversions.append(count_sequence_occurrences(act_arr, [2, 0, 2]) + count_sequence_occurrences(act_arr, [2, 1, 2]))
                            stress_level = []
                            act_arr = []
                            break            
    print("\n\n=======================================================")                  
    print("Plays = {}" .format(len(return_arr)))
    print("Wins = {}" .format(len(return_arr) - return_arr.count(-1.0)))
    wins = len(return_arr) - return_arr.count(-1.0)
    losses = len(return_arr) - wins
    print("Average return = {}" .format(np.average(list(filter(lambda a: a != -1.0, return_arr)))))
    print("Average stress level = {}" .format(np.average(sts_arr)))
    print("Total Aversions = {}" .format(np.sum(aversions)))
    print("Action Frequency = {}" .format(Counter(action_arr)))
    occurrences = count_string_occurrences(emotion_arr)
    emos = {'Anger': 0, 'Disgust': 0, 'Fear': 0, 'Guilt': 0, 'Joy': 0, 'Sadness': 0, 'Shame': 0}
    for string, count in occurrences.items():
        emos[string] = count
    print("Emotion count = {}" .format(emos))
    avg_r = np.average(list(filter(lambda a: a != -1.0, return_arr)))
    print("Score = {}" .format(minmax(((wins * avg_r) + (losses * -1))/len(return_arr))))

    grid = np.zeros((10,10))
    for i in agent_pos_arr:
        grid[i[0], i[1]] += 1 
    cv2.imwrite(f'runs/{args.gym_id}__{args.exp_name}/agent_roi.png', cv2.resize(cv2.normalize(grid, 0, 255, cv2.NORM_MINMAX), (256, 256), interpolation = cv2.INTER_AREA))

    import pandas as pd
    df = pd.DataFrame(app_data)
    df.to_csv('temp.csv')
    envs.close()