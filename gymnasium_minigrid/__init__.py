# -*- coding: utf-8 -*-

from gymnasium.envs.registration import register

# 注册自定义网格世界环境
register(
    # 环境的唯一ID，格式为 "Namespace/EnvName-vVersion"
    id="gymnasium_minigrid/GridWorld-v0",
    
    # 指向环境主类的入口点
    # 格式为 "path.to.module:ClassName"
    entry_point="gymnasium_minigrid.envs.grid_world:GridWorldEnv",
    
    # 每个回合的最大步数限制
    # 如果达到这个步数，环境会返回 truncated = True
    max_episode_steps=250,
)