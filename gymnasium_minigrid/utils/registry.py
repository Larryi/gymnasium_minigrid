"""
自定义函数注册表管理：管理 danger 和 movement 的用户自定义函数的短名注册。
"""
from typing import Callable, Dict

DANGER_REGISTRY: Dict[str, Callable] = {}
MOVEMENT_REGISTRY: Dict[str, Callable] = {}


def register_danger(name: str):
    def _decorator(fn: Callable):
        DANGER_REGISTRY[name] = fn
        return fn
    return _decorator


def register_movement(name: str):
    def _decorator(fn: Callable):
        MOVEMENT_REGISTRY[name] = fn
        return fn
    return _decorator


def list_dangers():
    return list(DANGER_REGISTRY.keys())


def list_movements():
    return list(MOVEMENT_REGISTRY.keys())
