import os
import argparse
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
import torch
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F


# Make vectorized environment function
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        # env = gym.make(gym_id)
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f'videos/{run_name}', episode_trigger = lambda x: x % 1000 == 0)
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

def certainity(logits): # Entropy
    """
    Computes the coping potential of an RL agent given the logits predicted by the model.

    Args:
        logits: A tensor of shape (batch_size, num_actions) containing the logits for each action.

    Returns:
        A tensor of shape (batch_size,) containing the coping potential for each example in the batch.
    """
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    norm = 1/(1+torch.exp(entropy))
    return norm

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        
        obs_size = np.array(envs.single_observation_space.shape).prod()
        self.appraisal_size = 1
        
        self.critic = nn.Sequential(
            layer_inint(nn.Linear(obs_size + self.appraisal_size, 64)),
            nn.Tanh(),
            layer_inint(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_inint(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_inint(nn.Linear(obs_size + self.appraisal_size, 64)),
            nn.Tanh(),
            layer_inint(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_inint(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x, appraisal):
        xe = torch.cat([x, appraisal], dim=-1)
        cr = self.critic(xe)
        napp = torch.tensor(certainity(cr).unsqueeze(1))
        return self.critic(xe), napp
    
    def get_action_and_value(self, x, appraisal, action=None):
        xe = torch.cat([x, appraisal], dim=-1)
        logits = self.actor(xe)
        napp = torch.tensor(certainity(logits).unsqueeze(1))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(xe), napp


# Arguments
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
                        help='Toggle leraning rate annealing')
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
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == "__main__":
    args = parse_args()
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

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    appraisals = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Try not to modify
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    next_appraisal = torch.tensor(certainity(next_obs).unsqueeze(1))

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value, app = agent.get_action_and_value(next_obs, next_appraisal)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # print(app.squeeze(1).shape)
            appraisals[step] = app.squeeze(1)

            # Ty not to modify: execute game and log data
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(terminated).to(device)
            # print(info)
            if(len(info)!=0):
                for item in info['final_info']:
                    if(item):
                        # print(item.keys())
                        if 'episode' in item.keys():
                            print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", item['episode']['r'], global_step)
                            writer.add_scalar("charts/episodic_length", item['episode']['l'], global_step)
                            break
                
        # bootstrap reward if not done with GAE
        with torch.no_grad():
            next_value, napp = agent.get_value(next_obs, next_appraisal)
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
                advantages = returns - values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_appraisal = appraisals.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, newapp = agent.get_action_and_value(
                    b_obs[mb_inds], b_appraisal[mb_inds].unsqueeze(1), b_actions.long()[mb_inds]
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

                # Entropy Loss
                entropy_loss = entropy.mean()

                # Appraisal Loss
                appraisal_targets = torch.tensor([0.5])
                appraisal_loss = F.mse_loss(torch.mean(newapp, 0), appraisal_targets, reduction='none').mean(-1, keepdim=True)
                # print(appraisal_loss)

                # Total Loss
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + 0.01 * appraisal_loss

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

        # try not to modify: record rewards for plotting

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("charts/value_loss", v_loss.item(), global_step)
        writer.add_scalar("charts/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("charts/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/appraisal_loss", appraisal_loss, global_step)
        print("SPS: ", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


