# -*- coding: utf-8 -*-
"""
config_utils.py

把字符串或对象转换成可调用对象（callable）。

- 传入callable，直接返回；
- 字符串形如（package.module.func），按完整路径导入；
- 否则按候选模块列表查找。

参数:
    name_or_obj: callable | str
    candidate_modules: 可选的模块路径列表

返回:
    callable

错误抛出:
    RuntimeError 如果无法解析
"""
import inspect
import importlib
from typing import Callable, Optional, Any
from functools import partial

from gymnasium_minigrid.utils import registry


def resolve_callable(spec: Any, kind: Optional[str] = None) -> Callable:
    """解析为可调用对象或绑定了参数的 callable。

    支持的 spec 形式：
      - callable -> 直接返回
      - str -> 在注册表中按短名查找
      - dict -> 必须包含 key `name`(短名或callable)，其余键将作为参数绑定到函数（使用 functools.partial）

    参数:
      kind: 可选，'danger' 或 'movement'，用于提示错误信息和限定注册表查找优先级。
    """
    if callable(spec):
        return spec

    # 字典形式 -> 需要绑定参数
    if isinstance(spec, dict):
        if 'name' not in spec:
            raise ValueError("传入 dict spec 必须包含 'name' 字段")
        name = spec['name']
        params = {k: v for k, v in spec.items() if k != 'name'}
        base = resolve_callable(name, kind=kind)
        if params:
            return partial(base, **params)
        return base

    # 字符串 -> 查注册表
    if isinstance(spec, str):
        # 确保聚合模块已导入以填充注册表（仅在注册表为空或首次调用时）
        try:
            # 导入聚合模块以触发注册（若已导入则无副作用）
            importlib.import_module("gymnasium_minigrid.utils.danger_func")
        except Exception:
            pass
        try:
            importlib.import_module("gymnasium_minigrid.utils.movement_func")
        except Exception:
            pass

        # 优先根据 kind 指定注册表
        if kind == 'danger':
            if spec in registry.DANGER_REGISTRY:
                return registry.DANGER_REGISTRY[spec]
        elif kind == 'movement':
            if spec in registry.MOVEMENT_REGISTRY:
                return registry.MOVEMENT_REGISTRY[spec]

        # 若未指定 kind 或在对应注册表中未找到，则在两个表中查找
        if spec in registry.DANGER_REGISTRY:
            return registry.DANGER_REGISTRY[spec]
        if spec in registry.MOVEMENT_REGISTRY:
            return registry.MOVEMENT_REGISTRY[spec]

        available = {
            "dangers": registry.list_dangers(),
            "movements": registry.list_movements(),
        }
        raise RuntimeError(
            f"无法解析函数名 '{spec}' 为可调用对象（kind={kind}）。可用选项: {available}"
        )

    raise TypeError("spec 必须为 callable、str 或 dict")
