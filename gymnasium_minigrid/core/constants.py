# -*- coding: utf-8 -*-

import numpy as np

# --- 基本对象类型定义 ---
# 整数来代表不同的对象
OBJECT_TO_IDX = {
    "empty": 0,   # 空地
    "wall": 1,    # 墙壁/障碍物
    "agent": 2,   # 我方 Agent
    "goal": 3,    # 终点
    "enemy": 4,   # 敌方 Agent
}

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}

# --- 渲染参数 ---
# 定义PyGame渲染的颜色 (RGB格式)
COLORS = {
    "background": (255, 255, 255), # 背景色 (白色)
    "wall": (40, 40, 40),          # 墙壁颜色 (深灰色)
    "agent": (0, 0, 255),       # 我方Agent颜色 (蓝色)
    "goal": (0, 255, 0),         # 终点颜色 (绿色)
    "enemy": (255, 0, 0),        # 敌方Agent颜色 (红色)
    "grid_lines": (200, 200, 200)  # 网格线颜色 (浅灰色)
}

# 危险区域的渲染颜色 (RGBA格式，最后一个值是透明度)
# 使用半透明的红色来叠加在危险区域上
DANGER_COLOR = (255, 0, 0, 128)

# 显示设置：渲染单元格的默认像素大小
CELL_SIZE = 16

# --- 动作定义 ---
# 定义Agent可执行的4个基本动作 (上, 下, 左, 右)
# 格式为 (row_change, col_change)
DIRECTIONS = np.array([
    [-1, 0],  # 0: 向上
    [1, 0],   # 1: 向下
    [0, -1],  # 2: 向左
    [0, 1],   # 3: 向右
])
