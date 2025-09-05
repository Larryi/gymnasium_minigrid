# -*- coding: utf-8 -*-

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame  # 导入pygame以备渲染器使用

from ..core.constants import OBJECT_TO_IDX, DIRECTIONS, CELL_SIZE
from ..core.config_utils import resolve_callable
from ..utils.movement_func import normalize_movement_spec
from ..core.agent import Agent
from ..rendering.pygame_renderer import PyGameRenderer

# --- 默认危险函数 ---
# 这是一个简单的默认实现，计算危险值。

def default_danger_func(grid_shape, enemy_locations, danger_radius):
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
    for r, c in enemy_locations:
        dist_sq = (h_indices - r) ** 2 + (w_indices - c) ** 2
        sigma = danger_radius / 2.0
        gauss = np.exp(-dist_sq / (2 * sigma ** 2))
        danger_map = np.maximum(danger_map, gauss.astype(np.float32))
    return danger_map


class GridWorldEnv(gym.Env):
    """
    一个高度可定制的离散网格世界环境。

    ### 观测空间
    观测是一个字典，包含:
    - `grid`: (np.ndarray) Agent视野范围内的网格对象信息。
    - `danger`: (np.ndarray) Agent视野范围内的危险值信息。

    ### 动作空间
    Agent有4个离散动作: 0 (上), 1 (下), 2 (左), 3 (右)。

    ### 回合结束 (Termination)
    - Agent到达终点
    - Agent撞上障碍物
    - Agent所在格子的危险值超过阈值

    ### 回合截断 (Truncation)
    - 达到最大步数限制
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        width=15,
        height=15,
        enemy_locations=None,
        obstacle_map=None,
        vision_radius=5,
        use_global_obs=False,
        danger_func=default_danger_func,
        danger_radius=4,
        danger_threshold=0.7,
        init_safe_threshold=0.3,
        max_steps=250,
        render_mode=None,
        debug_mode=False,
        fixed_agent_loc=None,
        fixed_goal_loc=None,
        enemy_movement=None
    ): 
        super().__init__()

        self.width = width
        self.height = height
        self.vision_radius = vision_radius
        self.use_global_obs = use_global_obs
        self.danger_func = danger_func
        self.danger_radius = danger_radius
        self.danger_threshold = danger_threshold
        self.init_safe_threshold = init_safe_threshold
        self._max_steps = max_steps

        # 记录固定位置参数
        self.fixed_agent_loc = fixed_agent_loc
        self.fixed_goal_loc = fixed_goal_loc
        self._step_count = 0

        # 保存enemy_locations模板，实际位置在reset时生成
        self._enemy_locations_template = enemy_locations if enemy_locations is not None else []
        self._enemy_locations = np.empty((0,2), dtype=int)
        # Agent 列表（每个敌人对应一个 Agent 实例）
        self._enemy_agents = []
        self._obstacle_map = obstacle_map
        self.enemy_movement = enemy_movement
        self._enemy_movement_callable = None
        # base_grid和danger_map推迟到reset时初始化

        # 如果用户在构造时通过短名或 dict 传入了函数规格，则立即解析为 callable
        try:
            if isinstance(self.danger_func, (str, dict)):
                self.danger_func = resolve_callable(self.danger_func, kind='danger')
        except Exception:
            raise
        try:
            if isinstance(self.enemy_movement, (str, dict)):
                self.enemy_movement = resolve_callable(self.enemy_movement, kind='movement')
        except Exception:
            raise

        # --- 定义观测空间和动作空间 ---
        self.action_space = spaces.Discrete(4)
        if self.use_global_obs:
            obs_shape = (self.height, self.width)
        else:
            obs_shape = (2 * self.vision_radius + 1, 2 * self.vision_radius + 1)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=max(OBJECT_TO_IDX.values()), shape=obs_shape, dtype=np.uint8),
            "danger": spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32),
            "rel_goal": spaces.Box(
                low=-max(self.width, self.height), high=max(self.width, self.height), shape=(2,), dtype=np.int32
            ),
        })
        self.debug_mode = debug_mode
        self.render_mode = render_mode
        self.renderer = None

    def _create_base_grid(self, obstacle_map, enemy_locations=None):
        """根据obstacle_map创建基础网格，如果为None则创建默认边墙。enemy_locations可选。"""
        grid = np.full((self.height, self.width), OBJECT_TO_IDX["empty"], dtype=np.uint8)
        if obstacle_map is None:
            # 默认在四周创建墙壁
            grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OBJECT_TO_IDX["wall"]
        else:
            # 使用用户提供的障碍物地图
            grid[obstacle_map == 1] = OBJECT_TO_IDX["wall"]
        # 将敌人放置在网格上
        if enemy_locations is not None:
            for r, c in enemy_locations:
                if 0 <= r < self.height and 0 <= c < self.width:
                    grid[r, c] = OBJECT_TO_IDX["enemy"]
        return grid

    def _get_random_empty_cell(self):
        """在网格中找到一个随机的、安全的、非障碍物、非敌人的格子。"""
        # 条件1: 必须是空地
        empty_mask = self._base_grid == OBJECT_TO_IDX["empty"]
        # 条件2: 必须是安全区域 (危险值低于一个阈值)
        safe_mask = self._danger_map < self.init_safe_threshold
        # 合并两个条件
        valid_cells = np.argwhere(np.logical_and(empty_mask, safe_mask))
        if len(valid_cells) == 0:
            raise RuntimeError(
                "无法在环境中找到任何安全的空地来放置Agent或目标。"
                "请检查您的敌人布局和危险半径设置，确保有足够的安全空间。"
            )
        idx = self.np_random.choice(len(valid_cells))
        return valid_cells[idx]

    def reset(self, seed=None, options=None, fixed_agent_loc=None, fixed_goal_loc=None):
        super().reset(seed=seed)

        # 1. 先生成无敌人的基础网格和危险图（用于安全采样敌人位置）
        self._enemy_locations = []
        self._base_grid = self._create_base_grid(self._obstacle_map, enemy_locations=None)
        self._danger_map = self.danger_func((self.height, self.width), np.empty((0,2), dtype=int), self.danger_radius)

        # 2. 采样/设置敌人实际位置
        used_cells = set()
        for loc in self._enemy_locations_template:
            if loc is None or (isinstance(loc, (tuple, list)) and (loc[0] is None or loc[1] is None)):
                cell = self._get_random_empty_cell()
                while tuple(cell) in used_cells:
                    cell = self._get_random_empty_cell()
                used_cells.add(tuple(cell))
                self._enemy_locations.append(cell)
            else:
                self._enemy_locations.append(np.array(loc))
        self._enemy_locations = np.array(self._enemy_locations, dtype=int) if self._enemy_locations else np.empty((0,2), dtype=int)

        # 3. 重新生成带敌人的网格和危险图
        self._base_grid = self._create_base_grid(self._obstacle_map, enemy_locations=self._enemy_locations)
        self._danger_map = self.danger_func((self.height, self.width), self._enemy_locations, self.danger_radius)

        # 4. 设置Agent和Goal位置（优先级：reset参数 > 环境构造参数 > 随机）
        agent_loc = None
        goal_loc = None
        if hasattr(self, 'fixed_agent_loc') and self.fixed_agent_loc is not None:
            agent_loc = self.fixed_agent_loc
        if hasattr(self, 'fixed_goal_loc') and self.fixed_goal_loc is not None:
            goal_loc = self.fixed_goal_loc
        if fixed_agent_loc is not None:
            agent_loc = fixed_agent_loc
        if fixed_goal_loc is not None:
            goal_loc = fixed_goal_loc
        # 统一输入格式：Agent/Goal 仅接受二元 tuple/list，且若包含 None 则表示随机
        if isinstance(agent_loc, (tuple, list)):
            if len(agent_loc) != 2:
                raise ValueError("fixed_agent_loc 必须是长度为2的 tuple/list 或 None")
            # 若任一元素为 None，则视为随机（等同于 None）
            if agent_loc[0] is None or agent_loc[1] is None:
                agent_loc = None
        if isinstance(goal_loc, (tuple, list)):
            if len(goal_loc) != 2:
                raise ValueError("fixed_goal_loc 必须是长度为2的 tuple/list 或 None")
            if goal_loc[0] is None or goal_loc[1] is None:
                goal_loc = None

        if agent_loc is not None:
            self._agent_location = np.array(agent_loc, dtype=int)
        else:
            self._agent_location = self._get_random_empty_cell()
        if goal_loc is not None:
            self._goal_location = np.array(goal_loc, dtype=int)
        else:
            self._goal_location = self._get_random_empty_cell()
        while np.array_equal(self._agent_location, self._goal_location):
            if goal_loc is not None:
                raise RuntimeError("Agent和Goal位置冲突，请检查fixed_agent_loc和fixed_goal_loc参数！")
            self._goal_location = self._get_random_empty_cell()

        self._step_count = 0

        # 解析 enemy_movement 为可调用对象或 per-enemy 列表
        self._enemy_movement_callable = None
        self._enemy_movement_callables = None
        if self.enemy_movement is not None:
            # 规范化传入规格（可能是单个、列表或字典）
            spec = normalize_movement_spec(self.enemy_movement, len(self._enemy_locations))
            if spec is None:
                self._enemy_movement_callable = None
            elif callable(spec) or isinstance(spec, str):
                # 全局 callable
                try:
                    self._enemy_movement_callable = resolve_callable(spec, kind='movement')
                except Exception:
                    if callable(spec):
                        self._enemy_movement_callable = spec
                    else:
                        raise
            else:
                # 按敌人列表解析每个元素
                callables = []
                for item in spec:
                    if item is None:
                        callables.append(None)
                        continue
                    if isinstance(item, str):
                        callables.append(resolve_callable(item, kind='movement'))
                    elif callable(item):
                        callables.append(item)
                    else:
                        raise TypeError("每个 enemy_movement 列表元素必须为 str 或 callable 或 None")
                self._enemy_movement_callables = callables

        # --- 创建 Agent 实例并分配 move_fn ---
        self._enemy_agents = []
        for i in range(len(self._enemy_locations)):
            pos = self._enemy_locations[i].copy()
            move_fn = None
            if self._enemy_movement_callables is not None:
                move_fn = self._enemy_movement_callables[i]
            elif self._enemy_movement_callable is not None:
                # 全局 callable
                move_fn = self._enemy_movement_callable
            agent = Agent(index=i, position=pos, team=None, move_fn=move_fn)
            self._enemy_agents.append(agent)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # 在应用主控Agent动作之前，先Agent.move
        if len(self._enemy_agents) > 0:
            try:
                moved_any = False
                for agent in self._enemy_agents:
                    # 让Agent根据其move_fn移动一次
                    moved = agent.move(self, self._step_count)
                    if moved:
                        moved_any = True
                if moved_any:
                    # 同步回 _enemy_locations
                    new_positions = np.array([ag.position for ag in self._enemy_agents], dtype=int)
                    self._enemy_locations = new_positions
                    # 更新网格和危险图
                    self._base_grid = self._create_base_grid(self._obstacle_map, enemy_locations=self._enemy_locations)
                    self._danger_map = self.danger_func((self.height, self.width), self._enemy_locations, self.danger_radius)
            except Exception as e:
                print(f"Warning: Agent-based enemy movement 调用失败: {e}")

        # 1. 计算新位置
        direction = DIRECTIONS[action]
        new_location = self._agent_location + direction

        terminated = False
        reward = -0.1  # 每步惩罚，鼓励尽快完成

        # 2. 检查边界和障碍物
        r, c = new_location
        if not (0 <= r < self.height and 0 <= c < self.width) or \
           self._base_grid[r, c] == OBJECT_TO_IDX["wall"]:
            terminated = True # 【终止】撞墙则终止
            reward = -5.0  # 撞墙重惩罚
            # 撞墙则位置不变

        else:
            prev_dist = np.linalg.norm(self._agent_location - self._goal_location, ord=1)
            self._agent_location = new_location
            curr_dist = np.linalg.norm(self._agent_location - self._goal_location, ord=1)

            # reward shaping: 靠近目标给予微小奖励
            reward += 0.25 * (prev_dist - curr_dist)

            # 4. 检查是否到达终点
            if np.array_equal(self._agent_location, self._goal_location):
                terminated = True #【终止】抵达目标则终止
                reward = 10.0  # 只给一个大正奖励

            # 5. 检查是否进入危险区域
            elif self._danger_map[r, c] > self.danger_threshold:
                terminated = True # 【终止】进入危险区域则终止
                reward = -5.0  # 进入危险区重惩罚

        # 6. 检查是否达到最大步数
        self._step_count += 1
        truncated = self._step_count >= self._max_steps #【截断】最大步数截断
        if truncated:
            reward -= 2.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        """根据全局或局部视野设置，获取当前的观测，并加入rel_goal。"""
        if self.use_global_obs:
            grid_obs = self._get_full_grid_obs()
            danger_obs = self._danger_map.copy()
        else:
            grid_obs, danger_obs = self._get_local_obs()

        rel_goal = self._goal_location - self._agent_location
        rel_goal = rel_goal.astype(np.int32)
        return {"grid": grid_obs, "danger": danger_obs, "rel_goal": rel_goal}

    def _get_full_grid_obs(self):
        """获取完整的全局网格观测。"""
        grid = self._base_grid.copy()
        grid[self._agent_location[0], self._agent_location[1]] = OBJECT_TO_IDX["agent"]
        grid[self._goal_location[0], self._goal_location[1]] = OBJECT_TO_IDX["goal"]
        return grid

    def _get_local_obs(self):
        """获取以Agent为中心的局部视野观测。"""
        r, c = self._agent_location
        rad = self.vision_radius
        
        # --- 提取局部网格 ---
        full_grid = self._get_full_grid_obs()
        padded_grid = np.pad(
            full_grid, 
            pad_width=rad, 
            mode='constant', 
            constant_values=OBJECT_TO_IDX["wall"]
        )
        grid_obs = padded_grid[r : r + 2 * rad + 1, c : c + 2 * rad + 1]

        # --- 提取局部危险地图 ---
        padded_danger = np.pad(
            self._danger_map,
            pad_width=rad,
            mode='constant',
            constant_values=1.0  # 视野外的区域视为最危险
        ).astype(np.float32)
        danger_obs = padded_danger[r : r + 2 * rad + 1, c : c + 2 * rad + 1]

        return grid_obs, danger_obs

    def _get_info(self):
        """获取辅助信息。"""
        dist = np.linalg.norm(self._agent_location - self._goal_location)
        return {
            "agent_location": self._agent_location,
            "goal_location": self._goal_location,
            "distance_to_goal": dist
        }

    def render(self):
        """渲染环境。"""
        if self.render_mode is None:
            gym.logger.warn("未指定渲染模式。默认使用human模式")
            self.render_mode = "human"

        if self.renderer is None:
            self.renderer = PyGameRenderer(
                self.render_mode,
                self.width,
                self.height,
                self._danger_map,
                cell_size=CELL_SIZE
            )
        # 每次渲染都传递最新danger_map，确保危险区背景随reset变化
        grid_for_render = self._get_full_grid_obs()
        return self.renderer.render(grid_for_render, danger_map=self._danger_map)

    def close(self):
        """关闭并清理渲染资源。"""
        if self.renderer:
            self.renderer.close()
            self.renderer = None

        self._base_grid = None
        self._danger_map = None
        self._agent_location = None
        self._goal_location = None
        self._step_count = None
        self._max_steps = None
