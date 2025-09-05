"""
聚合并注册敌人运动函数
"""
from .registry import register_movement

import numpy as np
from typing import Callable, List, Optional, Any


def static_movement(env, enemy_positions, step_count):
    """静止不动，返回 None 或相同数组"""
    return None


def random_walk_factory(p_move=1.0, rng=None):
    rng = rng or np.random

    def random_walk(env, enemy_positions, step_count):
        H, W = env.height, env.width
        new_pos = enemy_positions.copy()
        for i, (r, c) in enumerate(enemy_positions):
            if rng.random() > p_move:
                continue
            dirs = np.array([[-1,0],[1,0],[0,-1],[0,1]])
            order = rng.permutation(len(dirs))
            moved = False
            for idx in order:
                dr, dc = dirs[idx]
                nr, nc = int(r+dr), int(c+dc)
                if 0 <= nr < H and 0 <= nc < W and env._base_grid[nr, nc] != env._base_grid.max():
                    new_pos[i] = [nr, nc]
                    moved = True
                    break
            if not moved:
                new_pos[i] = [r, c]
        return new_pos

    return random_walk


def csv_follower_factory(trajectory_list: List[List[tuple]]):
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
    if spec is None:
        return None
    if callable(spec) or isinstance(spec, str):
        return spec
    if isinstance(spec, (list, tuple)):
        if len(spec) != num_enemies:
            raise ValueError(f"传入的 movement 列表长度应为 {num_enemies}（与num_enemies对应）")
        return list(spec)
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


# 注册常用短名
@register_movement("static")
def static(env, enemy_positions, step_count):
    return static_movement(env, enemy_positions, step_count)


@register_movement("random_walk")
def random_walk(env, enemy_positions, step_count):
    fn = random_walk_factory()
    return fn(env, enemy_positions, step_count)


__all__ = [
    "static", "random_walk", "static_movement", "random_walk_factory", "csv_follower_factory", "normalize_movement_spec",
]
