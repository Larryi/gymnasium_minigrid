import numpy as np
def circular_danger_func(grid_shape, enemy_locations, danger_radius):
    """
    以每个敌人为中心，生成危险场。

    参数:
        grid_shape (tuple): 网格的 (高度, 宽度)。
        enemy_locations (np.ndarray): 敌人位置的数组, shape=(N, 2)。
        danger_radius (int): 控制危险区域的半径（越大范围越广）。

    返回:
        np.ndarray: 一个与网格相同大小的浮点数组，表示危险值。
    """
    height, width = grid_shape
    danger_map = np.zeros((height, width), dtype=np.float32)
    if enemy_locations.shape[0] == 0:
        return danger_map

    h_indices, w_indices = np.indices((height, width))
    radius_sq = danger_radius ** 2
    for r, c in enemy_locations:
        r = int(r)
        c = int(c)
        # skip enemies outside the grid
        if r < 0 or r >= height or c < 0 or c >= width:
            continue
        dist_sq = (h_indices - r) ** 2 + (w_indices - c) ** 2
        mask = dist_sq <= radius_sq
        # set danger probability to 0.5 within the circle, do not accumulate on overlap
        danger_map[mask] = np.maximum(danger_map[mask], 0.5).astype(np.float32)
    return danger_map