# -*- coding: utf-8 -*-

import gymnasium as gym
import time

from gymnasium_minigrid.core.config_utils import resolve_callable
import numpy as np

def main():
    """主函数，用于创建、运行和测试环境。"""
    print("正在创建 'gymnasium_minigrid/GridWorld-v0' 环境...")

    # --- 环境配置 ---
    # 您可以在这里自定义环境的各种参数
    env_config = {
        "width": 40,
        "height": 40,
        "max_steps": 250,
        "enemy_locations": [(None, None), (None, None), (None, None), (None, None)],
        "fixed_agent_loc": (None, None),
        "fixed_goal_loc": (None, None),
        "danger_radius": 15,
        "danger_threshold": 0.7,
        "init_safe_threshold": 0.3,
        "use_global_obs": False,  # 设置为True以查看完整的地图
        "vision_radius": 10,
        "render_mode": "human",  # 设置为 "human" 以显示PyGame窗口
        "debug_mode": True,  # 设置为Debug模式，输出更多信息
        "danger_func": resolve_callable('circular_danger_func')
    }

    # 使用 gymnasium.make 创建环境实例，使用**env_config解包传参来覆盖默认参数
    env = gym.make("gymnasium_minigrid/GridWorld-v0", **env_config)

    print("环境创建成功！开始测试...")
    print("动作空间:", env.action_space)
    print("观测空间:", env.observation_space)

    # 运行10个回合的测试
    for episode in range(10):
        print(f"\n--- 回合 {episode + 1} ---")
        
        # 重置环境，获取初始观测
        obs, info = env.reset()
        
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        # 循环直到回合结束
        while not terminated and not truncated:
            # 渲染环境 (在 "human" 模式下，这会更新PyGame窗口)
            env.render()

            # 从动作空间中随机选择一个动作
            action = env.action_space.sample()
            # 或者：右下动作测试
            action = 3 if step_count % 2 == 0 else 1

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1

            if env.debug_mode:
                # 打印一些信息 (可以注释掉以获得更清晰的输出)
                print(f"步数: {step_count}, 动作: {action}, 奖励: {reward:.2f}")
                print(f"Agent 位置: {info['agent_location']}, 目标距离: {info['distance_to_goal']:.2f}")
            
            import matplotlib.pyplot as plt
            from gymnasium_minigrid.core.constants import OBJECT_TO_IDX

            # 设定colorbar范围
            grid_vmin, grid_vmax = min(OBJECT_TO_IDX.values()), max(OBJECT_TO_IDX.values())
            danger_vmin, danger_vmax = 0.0, 1.0

            # 初始化交互式模式和图窗，只在第一次时执行
            if episode == 0 and step_count == 1:
                plt.ion()
                fig, (ax1, ax2) = plt.subplots(1, 2)
                grid_img = ax1.pcolormesh(obs["grid"], cmap="hot", edgecolors='k', vmin=grid_vmin, vmax=grid_vmax)
                ax1.set_title("Grid")
                grid_cbar = plt.colorbar(grid_img, ax=ax1)
                grid_img.set_clim(grid_vmin, grid_vmax)
                danger_img = ax2.pcolormesh(obs["danger"], cmap="hot", edgecolors='k', vmin=danger_vmin, vmax=danger_vmax)
                ax2.set_title("Danger")
                danger_cbar = plt.colorbar(danger_img, ax=ax2)
                danger_img.set_clim(danger_vmin, danger_vmax)
            else:
                grid_img.set_array(obs["grid"].ravel())
                grid_img.set_clim(grid_vmin, grid_vmax)
                danger_img.set_array(obs["danger"].ravel())
                danger_img.set_clim(danger_vmin, danger_vmax)

            plt.pause(0.01)
            plt.draw()

        print(f"回合结束! 总步数: {step_count}, 总奖励: {total_reward:.2f}")
        if terminated:
            print("结束原因: Agent到达目标、撞墙或进入危险区域。")
        if truncated:
            print("结束原因: 达到最大步数限制。")

    # 关闭环境，释放资源
    env.close()
    print("\n测试完成。")

if __name__ == "__main__":
    main()
