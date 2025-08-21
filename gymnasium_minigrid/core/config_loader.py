# -*- coding: utf-8 -*-
"""
config_loader.py

加载YAML配置文件
"""
import os
import yaml

def load_config(config_path):
    """
    加载YAML配置文件，返回字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        if config:
            print(f"成功加载配置文件: {config_path}")
    return config

def get_config_for_run(config, algo=None):
    """
    提取对应算法参数
    """
    env_config = config.get('env', {})
    algo_config = config.get(algo, {}) if algo else {}
    return env_config, algo_config
