import datetime
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.agent_ppo import PPOAgent
from lib.buffer_ppo import PPOBuffer
from lib.utils import parse_args_ppo, make_env, log_video
import humanoid_with_bridge_env  # registra ambos envs
from gymnasium.wrappers import RecordVideo


def ppo_update(agent, optimizer, scaler, batch_obs, batch_actions, batch_returns, batch_old_log_probs, batch_adv,
               clip_epsilon, vf_coef, ent_coef):
    agent.train()
    optimizer.zero_grad()
    with torch.amp.autocast(str(device)):
        _, new_log_probs, entropies, new_values = agent.get_action_and_value(batch_obs, batch_actions)
        ratio = torch.exp(new_log_probs - batch_old_log_probs)
        kl = ((batch_old_log_probs - new_log_probs) / batch_actions.size(-1)).mean()
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(new_values.squeeze(1), batch_returns)
        entropy = entropies.mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), kl.item()


def train_phase(agent, optimizer, scheduler, scaler, args, env_id, n_epochs, run_id):
    current_dir = os.path.dirname(__file__)
    videos_dir = os.path.join(current_dir, f"videos_{env_id}", run_id)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", env_id, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(current_dir, f"logs_{env_id}", run_id)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # Create envs for this phase
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(env_id, reward_scaling=args.reward_scale) for _ in range(args.n_envs)]
    )
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    test_env = RecordVideo(
        test_env,
        video_folder=videos_dir,
        name_prefix=env_id,
        episode_trigger=lambda idx: idx % args.render_epoch == 0,
    )

    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape
    buffer = PPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)

    next_obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    next_terminated = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    next_truncated = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    reward_list = []
    episode_lengths = []
    step_counts = np.zeros(args.n_envs, dtype=int)
    best_mean_reward = -np.inf
    start_time = time.time()

    # Training loop for this phase
    for epoch in range(1, n_epochs + 1):
        # (Optionally adjust curriculum slopes here per phase)
        # Collect trajectories
        for _ in range(args.n_steps):
            obs = next_obs
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(obs)
            values = values.reshape(-1)
            next_obs, rewards, t, r, infos = envs.step(actions.cpu().numpy())
            step_counts += 1
            dones = t.astype(bool) | r.astype(bool)
            for i, done in enumerate(dones):
                if done:
                    episode_lengths.append(step_counts[i])
                    step_counts[i] = 0
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            reward_list.extend(rewards)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_terminated = torch.as_tensor(t, dtype=torch.float32, device=device)
            next_truncated = torch.as_tensor(r, dtype=torch.float32, device=device)
            buffer.store(obs, actions, rewards, values, next_terminated, next_truncated, logprobs)

        # Compute advantages and returns
        with torch.no_grad():
            next_values = agent.get_value(next_obs).reshape(1, -1)
            adv, ret = buffer.calculate_advantages(next_values, next_terminated.reshape(1,-1), next_truncated.reshape(1,-1))
        obs_buf, act_buf, logp_buf = buffer.get()
        obs_buf = obs_buf.view(-1, *obs_dim)
        act_buf = act_buf.view(-1, *act_dim)
        logp_buf = logp_buf.view(-1)
        adv = (adv.view(-1) - adv.mean()) / (adv.std() + 1e-8)
        ret = ret.view(-1)

        # Update policy
        dataset = args.n_steps * args.n_envs
        idxs = np.arange(dataset)
        for _ in range(args.train_iters):
            np.random.shuffle(idxs)
            for start in range(0, dataset, args.batch_size):
                batch = idxs[start:start+args.batch_size]
                loss, _, _, _, kl = ppo_update(
                    agent, optimizer, scaler,
                    obs_buf[batch], act_buf[batch], ret[batch], logp_buf[batch], adv[batch],
                    args.clip_ratio, args.vf_coef, args.ent_coef
                )
                if kl > args.target_kl:
                    break
            else:
                continue
            break

        mean_reward = float(np.mean(reward_list) / args.reward_scale)
        writer.add_scalar("reward/mean", mean_reward, epoch)
        reward_list.clear()
        print(f"{env_id} Phase Epoch {epoch}/{n_epochs}, mean reward: {mean_reward:.2f}")

        # Save models
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
        torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "last.pt"))

        if epoch % args.render_epoch == 0:
            log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch{epoch}.mp4"))
        scheduler.step()
        start_time = time.time()

    envs.close()
    test_env.close()
    writer.close()
    return os.path.join(checkpoint_dir, "best.pt")


if __name__ == "__main__":
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Initialize agent and training utilities once
    sample_env = make_env(args.env, reward_scaling=args.reward_scale)
    obs_dim = sample_env.observation_space.shape
    act_dim = sample_env.action_space.shape
    sample_env.close()

    agent = PPOAgent(obs_dim[0], act_dim[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
    scaler = torch.amp.GradScaler(str(device))

    # Split total epochs into soft and hard phases
    total = args.n_epochs
    soft = total // 2
    hard = total - soft
    phases = [
        ("HumanoidOnBridgeSoft-v0", soft),
        ("HumanoidOnBridge-v0", hard),
    ]

    # Run phases sequentially
    for env_id, ne in phases:
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt = train_phase(agent, optimizer, scheduler, scaler, args, env_id, ne, run_id)
        agent.load_state_dict(torch.load(ckpt, map_location=device))

    print("All training phases completed.")
