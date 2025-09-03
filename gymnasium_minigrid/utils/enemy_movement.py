# -*- coding: utf-8 -*-
"""
enemy_movement.py

预置的敌人运动函数

函数签名:
    fn(env, enemy_positions, step_count) -> new_enemy_positions (array-like Nx2) 或 None

- `env` 是环境对象，从中读取网格/障碍信息。
- `enemy_positions` 是Enemy位置，shape=(N,2)
- `step_count` 是当前环境的步数

实现:
- static_movement: 静止不动（返回 None 或相同位置）
- random_walk: 随机四向移动（避免墙壁），可选参数通过闭包传入
- csv_follower_factory: 从外部 CSV 或列表读取预定轨迹
"""

import numpy as np
from typing import Callable, List, Optional, Any


def static_movement(env, enemy_positions, step_count):
    """静止不动，返回 None 或相同数组"""
    return None


def random_walk_factory(p_move=1.0, rng=None):
    """
    返回一个随机游走函数，参数通过闭包指定。
    p_move: 每步移动的概率（0~1）
    rng: 可选 numpy.random.Generator
    """
    rng = rng or np.random

    def random_walk(env, enemy_positions, step_count):
        H, W = env.height, env.width
        new_pos = enemy_positions.copy()
        for i, (r, c) in enumerate(enemy_positions):
            if rng.random() > p_move:
                continue
            # 随机选择方向，上下左右或停
            dirs = np.array([[-1,0],[1,0],[0,-1],[0,1]])
            order = rng.permutation(len(dirs))
            moved = False
            for idx in order:
                dr, dc = dirs[idx]
                nr, nc = int(r+dr), int(c+dc)
                # 检查边界和墙壁
                if 0 <= nr < H and 0 <= nc < W and env._base_grid[nr, nc] != env._base_grid.max():
                    # 简单检查非墙即可移动
                    new_pos[i] = [nr, nc]
                    moved = True
                    break
            if not moved:
                new_pos[i] = [r, c]
        return new_pos

    return random_walk


def csv_follower_factory(trajectory_list: List[List[tuple]]):
    """
    trajectory_list: 长度为 N 的列表，N 为Enemy数。
    每个元素是一个 (T,2) 列表，表示在每一步的位置。如果某一步超出长度，则保持最后位置。

    返回一个函数，按step_count返回相应位置。
    """
    # 将轨迹标准化为 numpy arrays
    trajs = [np.array(t, dtype=int) for t in trajectory_list]

    def follower(env, enemy_positions, step_count):
        new_pos = []
        for i, traj in enumerate(trajs):
            if traj.shape[0] == 0:
                new_pos.append(enemy_positions[i])
            else:
                idx = min(step_count, traj.shape[0]-1)
                new_pos.append(traj[idx])
        return np.array(new_pos, dtype=int)

    return follower

def normalize_movement_spec(spec: Any, num_enemies: int):
    """
    将各种形式的enemy_movement规范化为长度为num_enemies的可调用对象列表或单个全局可调用。

    支持的spec类型：
      - None -> 返回 None
      - callable 或 str -> 返回该单一可调用（global）
      - callables/str 构成的List/tuple -> 返回 callable List， 长度为 num_enemies
      - dict index->callable -> 返回 包含None的List
    """
    if spec is None:
        return None

    # 单个 callable 或字符串（全局）
    if callable(spec) or isinstance(spec, str):
        return spec

    # 列表/元组 -> per-enemy
    if isinstance(spec, (list, tuple)):
        if len(spec) != num_enemies:
            raise ValueError(f"传入的 movement 列表长度应为 {num_enemies}（与num_enemies对应）")
        return list(spec)

    # 字典 mapping
    if isinstance(spec, dict):
        out = [None] * num_enemies
        for k, v in spec.items():
            if not isinstance(k, int):
                raise TypeError("字典键必须为敌人索引（int）")
            if k < 0 or k >= num_enemies:
                raise IndexError("字典键超出敌人数量范围")
            out[k] = v
        return out

    raise TypeError("无法解析的 enemy_movement 规格类型")
