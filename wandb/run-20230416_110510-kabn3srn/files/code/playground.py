import os
import argparse
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
import torch
import gym
# from gym.wrappers.record_video import RecordVideo

### Arguments ###

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', 
                        type=str, 
                        default=os.path.basename(__file__).rstrip('.py'), 
                        help='Name of Experiment')
    parser.add_argument('--gym-id',
                        type=str,
                        default='CartPole-v1',
                        help='ID of Gym Environment')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=2.5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Seed for the experiment')
    parser.add_argument('--total-timesteps',
                        type=int,
                        default=25000,
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
    parser.add_argument('--track',
                        type=lambda x:bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='If toggled, Weignts and biases will be used')
    parser.add_argument('--wandb-project-name',
                        type=str,
                        default='cleanRL',
                        help='W&B project name')
    parser.add_argument('--wandb-entity',
                        type=str,
                        default=None,
                        help='The entity of W&B project')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # print(args)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Weights and Biases
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Testing tensorboard
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # for i in range(100):
    #     writer.add_scalar("test_loss", i*2, global_step=i)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_det

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # # Environment sampling
    # env = gym.make(args.gym_id, render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.RecordVideo(env, 'videos', episode_trigger = lambda x: x % 100 == 0)
    # observation = env.reset()
    # # env.start_video_recorder()
    
    # for _ in range(200):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)  
    #     if terminated or truncated:
    #         observation =  env.reset()
    #         print(f"Episodic return: {info['episode']['r']}")
    # env.close()

    def make_env(gym_id):
        def thunk():
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.RecordVideo(env, 'videos', episode_trigger = lambda x: x % 100 == 0)
            return env
        return thunk
    
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)])
    observation = envs.reset()
    for _ in range(200):
        action = envs.action_space.sample()
        observation, reward, terminated, truncated, info = envs.step(action) 
        if terminated or truncated:
            print(f"Episodic return: {info['final_info'][0]['episode']['r']}")
    envs.close()
