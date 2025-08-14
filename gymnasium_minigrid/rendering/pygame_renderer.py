# -*- coding: utf-8 -*-

import pygame
import numpy as np
from ..core.constants import IDX_TO_OBJECT, COLORS, DANGER_COLOR, CELL_SIZE

class PyGameRenderer:
    """
    使用 PyGame 进行环境渲染的专用类。
    """
    def __init__(self, mode, width, height, danger_map, cell_size=CELL_SIZE):
        """
        初始化渲染器。

        参数:
            mode (str): 渲染模式 ('human' 或 'rgb_array')。
            width (int): 网格宽度。
            height (int): 网格高度。
            danger_map (np.ndarray): 完整的危险地图，用于渲染危险区域。
            cell_size (int): 每个网格单元的像素大小。
        """
        self.mode = mode
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.window_size = (width * cell_size, height * cell_size)
        
        self.window = None
        self.clock = None

        # 创建一个用于渲染危险区域的专用图层
        # 这个图层是半透明的，之后叠加到主屏幕上
        self.danger_surface = self._create_danger_surface(danger_map)

    def _create_danger_surface(self, danger_map):
        """根据危险地图创建一个半透明的表面。"""
        surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for r in range(self.height):
            for c in range(self.width):
                danger_value = danger_map[r, c]
                if danger_value > 0:
                    # 危险颜色 (R, G, B)
                    base_color = DANGER_COLOR[:3]
                    # 透明度 (Alpha)，与危险值成正比
                    alpha = int(DANGER_COLOR[3] * danger_value)
                    
                    cell_rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(surface, base_color + (alpha,), cell_rect)
        return surface

    def _init_pygame(self):
        """初始化PyGame窗口和时钟。"""
        pygame.init()
        if self.mode == "human":
            pygame.display.set_caption("GridWorld Environment")
            self.window = pygame.display.set_mode(self.window_size)
        else:  # rgb_array
            self.window = pygame.Surface(self.window_size)
        self.clock = pygame.time.Clock()

    def render(self, grid, danger_map=None):
        """
        渲染一帧。

        参数:
            grid (np.ndarray): 当前需要渲染的完整网格状态。
            danger_map (np.ndarray, optional): 最新危险地图。若提供则重建危险区图层。
        返回:
            np.ndarray or None: 如果模式是 'rgb_array'，返回图像数组；否则返回 None。
        """
        if self.window is None:
            self._init_pygame()
        if danger_map is not None:
            self.danger_surface = self._create_danger_surface(danger_map)

        # 1. 绘制背景
        self.window.fill(COLORS["background"])

        # 2. 绘制网格对象
        for r in range(self.height):
            for c in range(self.width):
                obj_idx = grid[r, c]
                obj_type = IDX_TO_OBJECT.get(obj_idx, "empty") # Fallback to empty
                # 只渲染非空对象
                if obj_type != "empty":
                    color = COLORS.get(obj_type)
                    if color:
                        cell_rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                        pygame.draw.rect(self.window, color, cell_rect)

        # 3. 叠加危险区域图层
        self.window.blit(self.danger_surface, (0, 0))

        # 4. 绘制网格线 (可选，但有助于观察)
        for r in range(self.height):
            pygame.draw.line(self.window, COLORS["grid_lines"], (0, r * self.cell_size), (self.window_size[0], r * self.cell_size))
        for c in range(self.width):
            pygame.draw.line(self.window, COLORS["grid_lines"], (c * self.cell_size, 0), (c * self.cell_size, self.window_size[1]))

        if self.mode == "human":
            pygame.display.flip()
            # 处理PyGame事件，例如关闭窗口
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            return None
        elif self.mode == "rgb_array":
            # 将PyGame表面转换为NumPy数组
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        """关闭PyGame窗口。"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None # 防止后续调用
