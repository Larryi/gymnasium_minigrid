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
import os


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
        "gymnasium_minigrid.utils"

    ]
    for mod_path in candidate_modules:
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, name_or_obj):
                obj = getattr(mod, name_or_obj)
                if callable(obj):
                    return obj
            # 如果模块是包且未直接包含属性，尝试扫描包下的子模块文件
            if hasattr(mod, '__path__'):
                try:
                    pkg_path = list(mod.__path__)[0]
                    for fname in os.listdir(pkg_path):
                        if not fname.endswith('.py'):
                            continue
                        if fname == '__init__.py':
                            continue
                        submodule_name = fname[:-3]
                        full_submodule = f"{mod_path}.{submodule_name}"
                        try:
                            submod = importlib.import_module(full_submodule)
                            if hasattr(submod, name_or_obj):
                                obj = getattr(submod, name_or_obj)
                                if callable(obj):
                                    return obj
                        except Exception:
                            continue
                except Exception:
                    pass
        except Exception:
            continue

    raise RuntimeError(
        f"无法解析函数名 '{name_or_obj}' 为可调用对象。"
    )
