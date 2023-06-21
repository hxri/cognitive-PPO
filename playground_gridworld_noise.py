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
import cv2

# Environment parameters
agent_view_size = 7
max_steps = 100
n_obstacles = 2
size = 10
agent_start_pos = None  # Dynamic start position

# Make vectorized environment function
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        # env = gym.make(gym_id)
        env = gym.make(gym_id,
                       render_mode="rgb_array",
                       max_steps=max_steps,
                       agent_view_size=agent_view_size,
                       n_obstacles=n_obstacles,
                       size=size,
                       agent_start_pos=agent_start_pos,
                       dynamic_wall=False,
                       dynamic_goal=True,
                       dynamic_obstacles=True,
                       moving_goal=False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                print("\n")
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger = lambda x: x % 10 == 0)
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

def appraisal_calc(obs=None, logits=None):
    a1 = motivational_relevance(obs)
    a2 = novelty(logits)
    a3 = certainity(logits)
    app = torch.stack((a1, a2, a3), -1)
    return app

def reward_with_app(base_rw, app):
    # new_rew = torch.where(base_rw == -1, base_rw, base_rw * app)
    new_rw = (base_rw.unsqueeze(-1) + 0.1 * torch.mean(app, dim=1, keepdim=True))
    # print(new_rw)
    return new_rw

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
            # nn.Linear(64*input_shape[0]*input_shape[1] + self.appraisal_size, 256),
            nn.Linear(64*input_shape[0]*input_shape[1], 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.actor = nn.Sequential(
            # nn.Linear(64*input_shape[0]*input_shape[1] + self.appraisal_size, 256),
            nn.Linear(64*input_shape[0]*input_shape[1], 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n)
        )

    def get_appraisal(self, x):
        xe = self.conv(x.permute(0, 3, 1, 2))
        app = appraisal_calc(x[:][..., 0], xe)
        # print(app)
        return app

    def get_value(self, x, appraisal):
        x1 = self.conv(x.permute(0, 3, 1, 2))
        # xe = torch.cat([x1, appraisal], dim=-1)
        # if(args.monitor_only):
        #     xe = x1
        # else:
        #     x2 = torch.mean((x1.unsqueeze(2) * appraisal.unsqueeze(1)), dim=2)
        #     xe = (x2 - torch.min(x2)) / (torch.max(x2) - torch.min(x2))
        xe = x1 + (0.01 * torch.rand(x1.shape))
        xe = (xe - torch.min(xe)) / (torch.max(xe) - torch.min(xe))
        cr = self.critic(xe)
        napp = appraisal_calc(x[:][..., 0], cr)
        return self.critic(xe), napp
            
    def get_action_and_value(self, x, appraisal, action=None):
        x1 = self.conv(x.permute(0, 3, 1, 2))
        # print([torch.max(x1), torch.min(x1)])
        # xe = torch.cat([x1, appraisal], dim=-1)
        # if(args.monitor_only):
        #     xe = x1
        # else:
        #     x2 = torch.mean((x1.unsqueeze(2) * appraisal.unsqueeze(1)), dim=2)
        #     xe = (x2 - torch.min(x2)) / (torch.max(x2) - torch.min(x2))
        # print(xe)
        xe = x1 + (0.01 * torch.rand(x1.shape))
        xe = (xe - torch.min(xe)) / (torch.max(xe) - torch.min(xe))
        logits = self.actor(xe)
        napp = appraisal_calc(x[:][..., 0], logits)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(xe), napp, x1



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
    parser.add_argument('--capture-video',
                        type=lambda x:bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='If toggled, captures the video of the agent')
    parser.add_argument('--monitor-only',
                        type=lambda x:bool(strtobool(x)),
                        default=False,
                        nargs='?',
                        const=True,
                        help='If toggled, enables appraisal monitoring only mode')
    
    # Algorithm specific
    parser.add_argument('--num-envs',
                        type=int,
                        default=4,
                        help='Number of parallel gym envs')
    parser.add_argument('--num-steps',
                        type=int,
                        default=128,
                        help='Number of steps to run each environment per policy rollout')
    parser.add_argument('--anneal-lr',
                        type=lambda x:bool(strtobool(x)),
                        default=True,
                        nargs='?',
                        const=True,
                        help='Toggle learning rate annealing')
    parser.add_argument('--gae',
                        type=lambda x:bool(strtobool(x)),
                        default=True,
                        nargs='?',
                        const=True,
                        help='Toggle General Advantage Estimation (GAE)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='Discount factor gamma')
    parser.add_argument('--gae-lambda',
                        type=float,
                        default=0.95,
                        help='Lambda for GAE')
    parser.add_argument('--num-minibatches',
                        type=int,
                        default=4,
                        help='Number of mini-batches')
    parser.add_argument('--update-epochs',
                        type=int,
                        default=4,
                        help='The k epochs to update policy')
    parser.add_argument('--norm-adv',
                        type=lambda x:bool(strtobool(x)),
                        default=True,
                        nargs='?',
                        const=True,
                        help='Toggle advantage normalization')
    parser.add_argument('--clip-coef',
                        type=float,
                        default=0.2,
                        help='The surrogate clipping coefficient')
    parser.add_argument('--clip-vloss',
                        type=lambda x:bool(strtobool(x)),
                        default=True,
                        nargs='?',
                        const=True,
                        help='Toggle the use of clipped value loss')
    parser.add_argument('--ent-coef',
                        type=float,
                        default=0.01,
                        help='The coefficient of entropy')
    parser.add_argument('--vf-coef',
                        type=float,
                        default=0.5,
                        help='The coefficient of value function')
    parser.add_argument('--max-grad-norm',
                        type=float,
                        default=0.5,
                        help='The maximum norm for gradient clipping')
    parser.add_argument('--target-kl',
                        type=float,
                        default=None,
                        help='The target KL divergence threshold')
    parser.add_argument('--early-stop',
                        type=int,
                        default=None,
                        help='The global step at which to stop the training')
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()
    # run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"{args.gym_id}__{args.exp_name}"

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
            settings=wandb.Settings(code_dir="."),
        )
    writer = SummaryWriter(f"runs/{run_name}")
    
    # Testing tensorboard
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" %('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_det

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only descrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Algorithm specific setup

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space['image'].shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    appraisals = torch.zeros((args.num_steps, args.num_envs) + (6,)).to(device)

    gc = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cp = torch.zeros((args.num_steps, args.num_envs)).to(device)
    anti = torch.zeros((args.num_steps, args.num_envs)).to(device)


    # Try not to modify
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]['image']).to(device)
    # print(envs.reset()[0]['image'])
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    next_appraisal = torch.tensor(agent.get_appraisal(next_obs))
    next_appraisal = torch.cat((next_appraisal, torch.tensor([[0], [0], [0], [0]]), torch.tensor([[0], [0], [0], [0]]), torch.tensor([[0], [0], [0], [0]])), dim=1)

    gc_prev = 0
    cp_prev = 0

    print(agent)

    for update in range(1, num_updates + 1):
        # Early stop
        if(args.early_stop is not None and global_step > args.early_stop):
            break

        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lrnow

        return_arr = -1
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done


            # Action logic
            with torch.no_grad():
                action, logprob, _, value, app, x1 = agent.get_action_and_value(next_obs, next_appraisal)
                # print(action)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            x1 = x1.numpy()[0]
            x1 = (x1-np.min(x1))/(np.max(x1)-np.min(x1))
            jj = x1.reshape((80,80))
            cv2.imwrite('temp.png', jj*255)

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            base_rw = torch.tensor(reward).to(device).view(-1)
            
            # print(info)
            # gc[step] = torch.tensor(info['goal_congruence'])
            # cp[step] = torch.tensor(info['coping_potential'])
            if('goal_congruence' in info):
                gc[step] = torch.tensor(info['goal_congruence'])
                cp[step] = torch.tensor(info['coping_potential'])
                gc_prev = gc[step]
                cp_prev = cp[step]
            else:
                gc[step] = gc_prev
                cp[step] = cp_prev
                print("Not found GC")
            
            # print(reward_with_app(base_rw, mot_rel))
            # if (global_step < 50000):
            #     rewards[step] = torch.tensor(reward).to(device).view(-1)
            # else:
            #     rewards[step] = reward_with_app(base_rw, mot_rel)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            anti[step] = (((rewards[step] * args.gamma -  values[step]) + 2) / 4)

            app = torch.cat((app, gc[step].unsqueeze(1), cp[step].unsqueeze(1), anti[step].unsqueeze(1)), dim=1)
            appraisals[step] = app
            # print(appraisals[step])

            # rewards[step] = torch.tensor(reward_with_app(base_rw, appraisals[step])).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs['image']).to(device), torch.Tensor(terminated).to(device)
            # print(info)
            if('final_info' in info):
                # print(info)
                for item in info['final_info']:
                    if(item):
                        # print(item)
                        if 'episode' in item.keys():
                            return_arr = item['episode']['r'].item()
                            print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                            # if(len(return_arr > 3)):
                            #     print(f"global_step={global_step}, episodic_return={np.average(return_arr)}")
                            # else:
                            #     print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                            # writer.add_scalar("charts/episodic_return", item['episode']['r'], global_step)
                            writer.add_scalar("charts/episodic_return", item['episode']['r'], global_step)
                            writer.add_scalar("charts/episodic_length", item['episode']['l'], global_step)
                            break
                 
        # bootstrap reward if not done with GAE
        with torch.no_grad():
            # print(next_obs)
            next_value, napp = agent.get_value(next_obs, appraisals[step])
            # mot_rel = torch.stack([item[0] for item in napp])
            next_value = next_value.reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps -1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                    # Incorporate motivational relevance into advantages
                    # print(advantages[t].shape)
                    # print(napp.shape)
                    # t1_prime = advantages[t].unsqueeze(1).expand_as(napp)
                    # result = t1_prime * napp
                    # advantages[t] = result.sum(dim=1)
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps -1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    
                    # # Incorporate motivational relevance into returns
                    # returns[t] = napp

                advantages = returns - values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space['image'].shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_appraisal = appraisals.reshape((-1,) + (6,))
        b_gc = gc.reshape(-1)
        b_cp = cp.reshape(-1)
        b_anti = anti.reshape(-1)


        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, newapp, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_appraisal[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().detach().cpu().numpy()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) **2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds]
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # Appraisal Loss
                newapp = torch.cat((newapp, b_gc[mb_inds].unsqueeze(1), b_cp[mb_inds].unsqueeze(1), b_anti[mb_inds].unsqueeze(1)), dim=1)
                # print(newapp)
                appraisal_targets = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                # print(newapp)
                # appraisal_loss = F.mse_loss(torch.mean(newapp, 0), appraisal_targets, reduction='none').mean(-1, keepdim=True)
                appraisal_loss = F.kl_div(torch.mean(newapp, 0), appraisal_targets, reduction='batchmean').mean(-1, keepdim=True)

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print("\nGS {} | R {:.3f} | lr {:.3e} | vL {:.3f} | pL {:.3f} | E {:.3f} | KL {:.3f} | cF {:.3f} | eVar {:.3f} | aL {:.3f} | SPS {}\n" .format(global_step,
                                                                                                                                                   return_arr, 
                                                                                                                                                    optimizer.param_groups[0]['lr'],
                                                                                                                                                    v_loss.item(),
                                                                                                                                                    pg_loss.item(),
                                                                                                                                                    entropy_loss.item(),
                                                                                                                                                    approx_kl.item(),
                                                                                                                                                    np.mean(clipfracs),
                                                                                                                                                    explained_var,
                                                                                                                                                    appraisal_loss.item(),
                                                                                                                                                    int(global_step / (time.time() - start_time)),))
        # Try not to modify: record rewards for plotting
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("charts/value_loss", v_loss.item(), global_step)
        writer.add_scalar("charts/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("charts/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/appraisal_loss", appraisal_loss, global_step)
        # print("SPS: ", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    envs.close()
    writer.close()

    # Save trained model
    torch.save(agent.state_dict(), f"runs/{run_name}/trained_agent.pt")


