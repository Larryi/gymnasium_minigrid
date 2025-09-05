# -*- coding: utf-8 -*-
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import sys
import os
from copy import deepcopy
from gymnasium_minigrid.core.config_loader import load_config, get_config_for_run
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    parsed = parser.parse_args()

    config = load_config(parsed.config)
    env_config, algo_config = get_config_for_run(config, algo="ppo")

    args = argparse.Namespace()

    defaults = {
        "env_id": "gymnasium_minigrid/GridWorld-v0",
        "exp_name": os.path.basename(__file__).rstrip(".py"),
        "seed": 1,
        "torch_deterministic": True,
        "cuda": False,
        "track": True,
        "test": False,
        "test_episodes": 10,
        "render": False,
        "num_envs": 8,
        "num_steps": 128,
        "num_minibatches": 4,
        "save_path": "trained_models/ppo_gridworld.pt",
        "load_path": None,
        "save_freq": 20000,
        "total_timesteps": 500000,
        "learning_rate": 2.5e-4,
        "async_envs": True,
    }
    for k, v in defaults.items():
        setattr(args, k, v)

    # algo参数
    for k, v in (algo_config or {}).items():
        setattr(args, k, v)
    # env参数
    for k, v in (env_config or {}).items():
        setattr(args, k, v)

    # 计算batch sizes
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


def make_env(env_id, seed, idx, run_name):
    """创建单个环境的辅助函数"""
    def thunk():
        # 支持通过全局变量传递env_config（由主程序注入）
        global GLOBAL_ENV_CONFIG
        env_config = deepcopy(GLOBAL_ENV_CONFIG) if 'GLOBAL_ENV_CONFIG' in globals() else {
            "width": 40,
            "height": 40,
            "enemy_locations": [(None, None), (None, None), (None, None)],
            "danger_radius": 15,
            "danger_threshold": 0.7,
            "init_safe_threshold": 0.3,
            "use_global_obs": False,
            "vision_radius": 10
        }
        # 记录环境参数到Log
        log_dir = os.path.join("runs", run_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "env_configs.log")
        with open(log_path, "a") as f:
            f.write(f"env_idx={idx}, seed={seed}, config={env_config}\n")

        env = gym.make(env_id, **env_config)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """初始化神经网络层的权重和偏置"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """定义处理自定义环境观测的PPO Agent"""
    def __init__(self, envs):
        super().__init__()
        
        obs_space = envs.single_observation_space
        grid_shape = obs_space["grid"].shape
        danger_shape = obs_space["danger"].shape
        
        max_grid_val = obs_space["grid"].high.max()
        self.grid_embedding = nn.Embedding(max_grid_val + 1, 8)  # 16->8

        # --- 小环境尽可能保留空间分辨率 ---
        self.grid_cnn = nn.Sequential(
            layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)),  # 40x40
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.danger_cnn = nn.Sequential(
            layer_init(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)),   # 40x40
            nn.ReLU(),
            layer_init(nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 动态推理flatten后特征维度，避免shape不匹配，并加上rel_goal
        with torch.no_grad():
            dummy_grid = torch.zeros(1, *grid_shape, dtype=torch.long)
            dummy_danger = torch.zeros(1, *danger_shape, dtype=torch.float32)
            dummy_rel_goal = torch.zeros(1, 2, dtype=torch.float32)
            embedded_grid = self.grid_embedding(dummy_grid).permute(0, 3, 1, 2)
            grid_features = self.grid_cnn(embedded_grid)
            danger_features = self.danger_cnn(dummy_danger.unsqueeze(1))
            total_features = grid_features.shape[1] + danger_features.shape[1] + 2

        self.critic = nn.Sequential(
            layer_init(nn.Linear(total_features, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(total_features, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(self._get_common_features(x))

    def get_action_and_value(self, x, action=None):
        features = self._get_common_features(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)

    def _get_common_features(self, x):
        # x是一个字典: {"grid": ..., "danger": ..., "rel_goal": ...}
        grid_obs = x["grid"].long()
        danger_obs = x["danger"].unsqueeze(1) # 添加通道维度
        rel_goal = x["rel_goal"].float()  # (N, 2)

        embedded_grid = self.grid_embedding(grid_obs).permute(0, 3, 1, 2)
        grid_features = self.grid_cnn(embedded_grid)
        danger_features = self.danger_cnn(danger_obs)

        # 拼接所有特征
        combined_features = torch.cat([grid_features, danger_features, rel_goal], dim=1)
        return combined_features


if __name__ == "__main__":
    args = parse_args()

    # --- 注入全局env_config供make_env使用 ---
    global GLOBAL_ENV_CONFIG
    GLOBAL_ENV_CONFIG = {}
    for k in [
        "width", "height", "enemy_locations", "danger_radius", "danger_threshold", "init_safe_threshold",
        "use_global_obs", "vision_radius", "max_steps", "render_mode", "debug_mode", "fixed_agent_loc", "fixed_goal_loc", "obstacle_map", "danger_func", "enemy_movement"
    ]:
        if hasattr(args, k):
            GLOBAL_ENV_CONFIG[k] = getattr(args, k)

    try:
        from gymnasium_minigrid.core.config_utils import resolve_callable
        if "danger_func" in GLOBAL_ENV_CONFIG and GLOBAL_ENV_CONFIG["danger_func"] is not None:
            GLOBAL_ENV_CONFIG["danger_func"] = resolve_callable(GLOBAL_ENV_CONFIG["danger_func"], kind='danger')
        if "enemy_movement" in GLOBAL_ENV_CONFIG:
            GLOBAL_ENV_CONFIG["enemy_movement"] = resolve_callable(GLOBAL_ENV_CONFIG["enemy_movement"], kind='movement')
    except Exception:
        pass

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    args.save_path = os.path.join("trained_models", run_name, "model.pt")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "mps")

    if args.test:
        # 测试模式：单环境评估
        print("进入测试模式...")
        env = make_env(args.env_id, args.seed, 0, run_name)()
        agent = Agent(type('DummyVec', (), {'single_observation_space': env.observation_space, 'single_action_space': env.action_space}))
        agent.to(device)
        assert args.load_path is not None and os.path.exists(args.load_path), "测试模式必须指定已训练模型的 --load-path"
        checkpoint = torch.load(args.load_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        episode_returns = []
        episode_lengths = []
        for ep in range(args.test_episodes):
                obs, _ = env.reset(seed=args.seed+ep)
                obs_grid = torch.tensor(obs['grid'], dtype=torch.uint8).unsqueeze(0).to(device)
                obs_danger = torch.tensor(obs['danger']).float().unsqueeze(0).to(device)
                obs_rel_goal = torch.tensor(obs['rel_goal']).float().unsqueeze(0).to(device)
                done = False
                ep_ret = 0.0
                ep_len = 0
                while not done:
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value({"grid": obs_grid, "danger": obs_danger, "rel_goal": obs_rel_goal})
                    action_np = action.cpu().numpy()[0]
                    obs, reward, terminated, truncated, info = env.step(action_np)
                    done = terminated or truncated
                    ep_ret += reward
                    ep_len += 1
                    obs_grid = torch.tensor(obs['grid'], dtype=torch.uint8).unsqueeze(0).to(device)
                    obs_danger = torch.tensor(obs['danger']).float().unsqueeze(0).to(device)
                    obs_rel_goal = torch.tensor(obs['rel_goal']).float().unsqueeze(0).to(device)
                    if args.render:
                        env.render()
                print(f"[Test] Episode {ep+1}/{args.test_episodes} | Return: {ep_ret:.2f} | Length: {ep_len}")
                episode_returns.append(ep_ret)
                episode_lengths.append(ep_len)
        print(f"[Test] 平均回报: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f} | 平均长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        env.close()
        exit(0)

    # 训练模式
    # 设置TensorBoard
    if args.track:
        writer = SummaryWriter(f"runs/{run_name}")
        # 记录所有超参数到TensorBoard
        hp_table = "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in sorted(vars(args).items())])
        writer.add_text("hyperparameters", hp_table)

    # 创建并行环境
    env_fns = [make_env(args.env_id, args.seed + i, i, run_name) for i in range(args.num_envs)]
    if args.async_envs:
        print(f"使用 AsyncVectorEnv 创建 {args.num_envs} 个并行环境")
        try:
            envs = gym.vector.AsyncVectorEnv(env_fns)
        except Exception as e:
            print(f"AsyncVectorEnv 创建失败: {e}")
            print("回退到 SyncVectorEnv")
            envs = gym.vector.SyncVectorEnv(env_fns)
    else:
        print(f"使用 SyncVectorEnv 创建 {args.num_envs} 个环境")
        envs = gym.vector.SyncVectorEnv(env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # --- 加载模型 ---
    start_step = 0
    if args.load_path and os.path.exists(args.load_path):
        print(f"加载模型: {args.load_path}")
        checkpoint = torch.load(args.load_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['global_step']
        print(f"模型加载成功，从第 {start_step} 步继续训练。")


    # --- PPO 存储设置 ---
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space['grid'].shape).to(device)
    dangers = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space['danger'].shape).to(device)
    rel_goals = torch.zeros((args.num_steps, args.num_envs, 2), dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # --- 训练循环 ---
    global_step = start_step
    # 记录本次训练的起始step和时间（用于SPS和耗时统计）
    train_start_step = global_step
    train_start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    
    # 预分配tensor以减少内存分配开销
    next_obs_grid = torch.zeros((args.num_envs,) + envs.single_observation_space['grid'].shape, dtype=torch.uint8, device=device)
    next_obs_danger = torch.zeros((args.num_envs,) + envs.single_observation_space['danger'].shape, dtype=torch.float32, device=device)
    next_obs_rel_goal = torch.zeros((args.num_envs, 2), dtype=torch.float32, device=device)
    
    # 初始化观测
    next_obs_grid.copy_(torch.from_numpy(next_obs['grid']).to(device))
    next_obs_danger.copy_(torch.from_numpy(next_obs['danger']).to(device))
    next_obs_rel_goal.copy_(torch.from_numpy(next_obs['rel_goal']).to(device))
    
    next_done = torch.zeros(args.num_envs, device=device)
    num_updates = args.total_timesteps // args.batch_size

    # 维护每个env的累计reward和length
    episode_rewards = [0.0 for _ in range(args.num_envs)]
    episode_lengths = [0 for _ in range(args.num_envs)]
    # 新增：维护所有episode回报的全局列表，用于TensorBoard可视化
    all_episode_rewards = []

    for update in range(1, num_updates + 1):
        # 学习率线性衰减
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Rollout/采样阶段 ---
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step].copy_(next_obs_grid)
            dangers[step].copy_(next_obs_danger)
            rel_goals[step].copy_(next_obs_rel_goal)
            dones[step].copy_(next_done)

            current_obs = {"grid": next_obs_grid, "danger": next_obs_danger, "rel_goal": next_obs_rel_goal}
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(current_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            rewards[step] = torch.from_numpy(reward).float().to(device)
            # 使用copy_避免重新分配内存
            next_obs_grid.copy_(torch.from_numpy(next_obs['grid']).to(device))
            next_obs_danger.copy_(torch.from_numpy(next_obs['danger']).to(device))
            next_obs_rel_goal.copy_(torch.from_numpy(next_obs['rel_goal']).to(device))
            next_done.copy_(torch.from_numpy(done).float().to(device))

            # --- 手动统计每个env的episode ---
            for i in range(args.num_envs):
                episode_rewards[i] += reward[i]
                episode_lengths[i] += 1
                if done[i]:
                    ep_rew = episode_rewards[i]
                    ep_len = episode_lengths[i]
                    print(f"[Episode End] Env#{i} | global_step={global_step} | episodic_return={ep_rew:.2f} | episodic_length={ep_len}")
                    if args.track:
                        writer.add_scalar("charts/episodic_return", ep_rew, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)
                    # 新增：统计全局平均
                    all_episode_rewards.append(ep_rew)
                    if args.track and len(all_episode_rewards) > 0:
                        writer.add_scalar("charts/mean_reward_cumulative", np.mean(all_episode_rewards), global_step)
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0

        # --- GAE 和 价值目标计算 ---
        with torch.no_grad():
            next_value = agent.get_value({"grid": next_obs_grid, "danger": next_obs_danger, "rel_goal": next_obs_rel_goal}).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # --- 学习/更新阶段 ---
        # 直接reshape而不创建新的tensor
        b_obs = obs.view(-1, *envs.single_observation_space['grid'].shape)
        b_dangers = dangers.view(-1, *envs.single_observation_space['danger'].shape)
        b_rel_goals = rel_goals.view(-1, 2)
        b_logprobs = logprobs.view(-1)
        b_actions = actions.view(-1, *envs.single_action_space.shape)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = values.view(-1)
        # 预先分配索引数组避免重复创建
        b_inds = torch.arange(args.batch_size, device=device)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # 使用torch.randperm更高效
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = {"grid": b_obs[mb_inds], "danger": b_dangers[mb_inds], "rel_goal": b_rel_goals[mb_inds]}
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 策略损失 (Policy Loss)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 价值损失 (Value Loss)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # 熵损失 (Entropy Loss)
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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
        # --- 计算并记录SPS ---
        # 只统计本次训练的SPS
        sps = int((global_step - train_start_step) / (time.time() - train_start_time))

        # --- 控制台和Tensorboard日志: 每次更新 ---
        print(f"\n--- 更新 {update}/{num_updates} (SPS: {sps}) ---")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"价值损失: {v_loss.item():.4f}, 策略损失: {pg_loss.item():.4f}, 熵: {entropy_loss.item():.4f}")
        print(f"解释方差: {explained_var:.4f}, KL散度: {approx_kl.item():.4f}")
        if args.track:
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("charts/explained_variance", explained_var, global_step)

            # --- 自动保存模型 ---
            if args.save_freq > 0 and global_step % args.save_freq < args.batch_size:
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                save_payload = {
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step
                }
                # 保存带步数后缀的模型
                base, ext = os.path.splitext(args.save_path)
                save_name = f"{base}_step{global_step}{ext}" if args.save_freq > 0 else args.save_path
                torch.save(save_payload, save_name)
                print(f"[自动保存] 模型已保存到: {save_name}")

    # --- 保存最终模型 ---
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    save_payload = {
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }
    torch.save(save_payload, args.save_path)
    print(f"模型已保存到: {args.save_path}")

    # 输出训练总耗时
    train_time = time.time() - train_start_time
    print(f"训练总耗时: {train_time:.1f} 秒 ({train_time/60:.2f} 分钟)")

    # 安全关闭环境
    try:
        envs.close()
        print("环境已正常关闭")
    except Exception as e:
        print(f"关闭环境时出现警告: {e}")
    
    if args.track:
        writer.close()
