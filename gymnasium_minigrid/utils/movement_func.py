"""
聚合并注册敌人运动函数
"""
from .registry import register_movement

import numpy as np
from ..core.constants import OBJECT_TO_IDX
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


def correlated_random_walk_factory(angular_speed=0.2, noise_scale=0.3, rng=None, seed=None):
    """
    创建平滑随机游走函数
    """
    if rng is None:
        rng = np.random.RandomState(seed) if seed is not None else np.random

    def correlated_random_walk(env, enemy_positions, step_count):
        H, W = env.height, env.width
        new_pos = enemy_positions.copy()
        # 对每个敌人基于其索引与初始位置确定一个固定的 base_phase，以便轨迹多样且可重现
        for i, (r, c) in enumerate(enemy_positions):
            # 使用位置与索引生成一个伪随机但确定的基础角度（度数 -> 弧度）
            base = ((r + 1) * 31 + (c + 1) * 97 + i * 13) % 360
            base_phase = (base / 180.0) * np.pi

            # 角度随步数线性演化并叠加小幅高斯噪声
            noise = noise_scale * (rng.randn() if hasattr(rng, 'randn') else np.random.randn())
            angle = base_phase + angular_speed * step_count + noise

            # 将角度映射到最近的格子方向（通过 cos/sin 的四舍五入）
            dx = int(np.round(np.cos(angle)))
            dy = int(np.round(np.sin(angle)))
            # 转换为行列变化（行号向下增加）
            dr = -dy
            dc = dx

            nr, nc = int(r + dr), int(c + dc)
            # 若前方格合法则移动
            if 0 <= nr < H and 0 <= nc < W and env._base_grid[nr, nc] != OBJECT_TO_IDX["wall"]:
                new_pos[i] = [nr, nc]
            else:
                # 否则尝试回退到相邻的合法格
                candidates = [
                    (int(r - 1), int(c)),
                    (int(r + 1), int(c)),
                    (int(r), int(c - 1)),
                    (int(r), int(c + 1)),
                ]
                moved = False
                for (tr, tc) in candidates:
                    if 0 <= tr < H and 0 <= tc < W and env._base_grid[tr, tc] != OBJECT_TO_IDX["wall"]:
                        new_pos[i] = [tr, tc]
                        moved = True
                        break
                # 若无可行格，则保持原地不动
                if not moved:
                    new_pos[i] = [r, c]
        return new_pos

    return correlated_random_walk


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


# 注册新的平滑随机游走
@register_movement("correlated_random_walk")
def correlated_random_walk(env, enemy_positions, step_count):
    fn = correlated_random_walk_factory()
    return fn(env, enemy_positions, step_count)


@register_movement("chase_agent")
def chase_agent(env, enemy_positions, step_count):
    """每个敌人朝当前 Agent 位置靠近，选择能最小化距离的合法动作（避免墙与与其他敌人冲突）。"""
    H, W = env.height, env.width
    agent_r, agent_c = int(env._agent_location[0]), int(env._agent_location[1])
    new_pos = enemy_positions.copy()

    # 当前被占据的位置（所有敌人的当前位置）
    occupied = set((int(r), int(c)) for (r, c) in enemy_positions)
    # 已被前面敌人选中的目标位置（避免同一帧冲突）
    reserved = set()
    wall_idx = OBJECT_TO_IDX["wall"]

    for i, (r0, c0) in enumerate(enemy_positions):
        r, c = int(r0), int(c0)

        # 候选动作：原地，上，下，左，右
        candidates = [
            (r, c),
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ]

        valid = []
        for (nr, nc) in candidates:
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if env._base_grid[nr, nc] == wall_idx:
                continue
            # 避免移动到当前被其他敌人占据的位置（允许留在自己当前位置）
            if (nr, nc) in occupied and (nr, nc) != (r, c):
                continue
            # 避免已被前面敌人预占的位置
            if (nr, nc) in reserved:
                continue
            valid.append((nr, nc))

        if not valid:
            chosen = (r, c)
        else:
            valid_sorted = sorted(valid, key=lambda p: (abs(p[0] - agent_r) ** 2 + abs(p[1] - agent_c) ** 2))
            best = valid_sorted[0]

            # 仅在不会增加与 agent 距离的情况下移动，否则保持原地
            curr_dist = (r - agent_r) ** 2 + (c - agent_c) ** 2
            best_dist = (best[0] - agent_r) ** 2 + (best[1] - agent_c) ** 2
            if best_dist <= curr_dist:
                chosen = best
            else:
                chosen = (r, c)

        new_pos[i] = [chosen[0], chosen[1]]

        # 更新占用/预占集合，避免随后敌人冲突或走到刚被移动到的位置
        occupied.discard((r, c))
        occupied.add(chosen)
        reserved.add(chosen)

    return np.array(new_pos, dtype=int)


__all__ = [
    "static", "random_walk", "static_movement", "random_walk_factory", "csv_follower_factory", "normalize_movement_spec", "correlated_random_walk", "chase_agent",
]
