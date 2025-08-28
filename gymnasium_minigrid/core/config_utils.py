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
import importlib


def resolve_callable(name_or_obj, candidate_modules=None):
    """
    解析为可调用对象
    """
    # 已是 callable，直接返回
    if callable(name_or_obj):
        return name_or_obj
    if not isinstance(name_or_obj, str):
        raise TypeError("resolve_callable 应输入 callable 或 str")

    # 完整 dotted path
    if "." in name_or_obj:
        module_path, func_name = name_or_obj.rsplit('.', 1)
        try:
            mod = importlib.import_module(module_path)
            obj = getattr(mod, func_name)
            if callable(obj):
                return obj
        except Exception:
            pass

    # 候选模块列表
    candidate_modules = candidate_modules or [
        "gymnasium_minigrid.core",
        "gymnasium_minigrid.envs",
        "gymnasium_minigrid.rendering",
        "gymnasium_minigrid",
        "utils",
    ]
    for mod_path in candidate_modules:
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, name_or_obj):
                obj = getattr(mod, name_or_obj)
                if callable(obj):
                    return obj
        except Exception:
            continue

    raise RuntimeError(f"无法解析函数名 '{name_or_obj}' 为callable。")
