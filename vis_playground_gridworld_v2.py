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
from torch.autograd import Variable


# Environment parameters
agent_view_size = 7
max_steps = 1000
n_obstacles = 14
size = 21
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
                       moving_goal=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

# Weight and Bias intialization for PPO
def layer_inint(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Weight and Bias intialization for PPO
def layer_inint(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            Attention(64),
            nn.Flatten()
        )

        input_shape = (agent_view_size + 3, agent_view_size + 3)
        self.critic = nn.Sequential(
            nn.Linear(64*input_shape[0]*input_shape[1], 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(64*input_shape[0]*input_shape[1], 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n)
        )

    def get_value(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        x = self.conv(x.permute(0, 3, 1, 2))

        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


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

    # Try not to modify
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]['image']).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    return_arr = []
    for update in range(1, num_updates + 1):
        for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    # print(action)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Ty not to modify: execute game and log data
                next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                # print(np.average(reward))
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs['image']).to(device), torch.Tensor(terminated).to(device)
                # print(info)
                time.sleep(.1)
                if('final_info' in info):
                    for item in info['final_info']:
                        if(item):
                            # print(item)
                            if 'episode' in item.keys():
                                return_arr.append(item['episode']['r'])
                                # print(f"global_step={global_step}, episodic_return={np.average(return_arr)}")
                                print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                                break                    
    print("\n\n=======================================================")                  
    print("Plays = {}" .format(len(return_arr)))
    print("Wins = {}" .format(len(return_arr) - return_arr.count(-1.0)))
    print("Average return = {}" .format(np.average(list(filter(lambda a: a != -1.0, return_arr)))))
    envs.close()