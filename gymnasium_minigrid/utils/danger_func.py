"""
聚合并注册危险函数
"""
from .registry import register_danger

import numpy as np


@register_danger("circular")
def circular(grid_shape, enemy_locations, danger_radius):
    """以每个敌人为中心，生成危险场"""
    height, width = grid_shape
    danger_map = np.zeros((height, width), dtype=np.float32)
    if getattr(enemy_locations, 'shape', (0,))[0] == 0:
        return danger_map

    h_indices, w_indices = np.indices((height, width))
    radius_sq = danger_radius ** 2
    for r, c in enemy_locations:
        r = int(r)
        c = int(c)
        if r < 0 or r >= height or c < 0 or c >= width:
            continue
        dist_sq = (h_indices - r) ** 2 + (w_indices - c) ** 2
        mask = dist_sq <= radius_sq
        danger_map[mask] = np.maximum(danger_map[mask], 0.5).astype(np.float32)
    return danger_map


__all__ = ["circular"]
