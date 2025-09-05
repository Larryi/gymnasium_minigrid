# -*- coding: utf-8 -*-
"""
简单的 Agent 表示（用于Enemy）

属性：index, position, team, move_fn

fn(env, enemy_positions, step_count) -> new_positions 或 None
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Callable, Any


@dataclass
class Agent:
    index: int
    position: np.ndarray
    team: Optional[str] = None
    move_fn: Optional[Callable] = None

    def as_tuple(self):
        return (int(self.position[0]), int(self.position[1]))

    def move(self, env: Any, step_count: int):
        """调用 move_fn（若存在），并在成功时更新 self.position。

        返回: True 如果位置发生更新。
        """
        if self.move_fn is None:
            return False

        try:
            res = self.move_fn(env, np.array([self.position]), step_count)
        except TypeError:
            # 允许一些更宽松的签名：move_fn(env, all_enemy_positions, step_count)
            try:
                res = self.move_fn(env, env._enemy_locations.copy(), step_count)
            except Exception as e:
                # 报告但不抛出
                print(f"Warning: Agent.move inner move_fn 调用失败: {e}")
                return False
        except Exception as e:
            print(f"Warning: Agent.move 调用失败: {e}")
            return False

        if res is None:
            return False

        arr = np.array(res, dtype=int)
        # 单个位置
        if arr.ndim == 1 and arr.size == 2:
            self.position = arr
            return True
        # 返回一组位置，取第一个
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
            self.position = arr[0]
            return True

        print("Warning: Agent.move 收到不合法的返回形状")
        return False
