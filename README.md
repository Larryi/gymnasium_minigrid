![](https://img.shields.io/badge/Env-gymnasium_minigrid-blue)

# gymnasium_minigrid: 灵活可扩展的自定义网格环境

`gymnasium_minigrid` 是一个离散网格地图，可以配置局部/全局视野、危险系数、敌人、目标等多种元素的自定义 RL 环境，兼容 Gymnasium (Gym) 接口，适用于强化学习算法的研究与测试，可用于智能体导航、避障、探索等任务。

## 特性

- 自定义地图大小
- 局部/全局视野可切换
- 危险区、敌人、目标等多元素可配置
- 支持自定义危险函数、Agent/目标/敌人初始位置固定或随机
- 可视化（可渲染 PyGame 窗口，调试时可展示 matplotlib 热力图）
- 兼容 Gymnasium，便于集成主流 RL 算法

## 接口与参数

### 网格子成分

环境中的 `grid` 的每个格子为整数编码，含义如下：

| 数值 | 含义         | 解释                                                     |
| ---- | ------------ | -------------------------------------------------------- |
| 0    | 空地         | 可通行区域，Agent 可自由移动                             |
| 1    | 墙体或障碍物 | 不可通行区域，Agent 不能进入，若触碰会被传送回上一个位置 |
| 2    | 敌人         | 敌人实体，依此生成危险区域                               |
| 3    | 目标         | 任务目标，Agent 到达此处则认为完成任务                   |
| 4    | Agent 本体   | 己方 Agent 实体                                          |

具体编码规则见： `gymnasium_minigrid/core/constants.py` 的 `OBJECT_TO_IDX`。

说明：

- 敌人可以设定为多个
- 敌人、目标、Agent 本体的位置可固定或随机生成，若为随机生成则会根据安全性和网格可用性自动选取。
- 观测空间、动作空间、奖励等可通过环境参数自定义。

**状态空间**：字典中含 `grid` (地图)、`danger` (危险概率)、`rel_goal` (相对目标坐标)

**动作空间**：离散，0=上、1=下、2=左、3=右。

**奖励与终止**：

- 每步基础奖励 -0.1
- 靠近目标有微小正奖励（曼哈顿距离减少 × 0.25）
- 撞墙或进入危险区：-5.0 并终止
- 到达目标：+10.0 并终止
- 达到最大步数截断：-2.0 并截断

- 终止条件为：到达目标、撞墙障碍物、进入危险区（超出危险阈值）
- 截断条件为：达到最大步数

**主要参数表：**

| 参数名                | 类型       | 说明                                                                                       |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| `width`/`height`      | int        | 地图宽/高，支持自定义尺寸（如 40x40）                                                      |
| `max_steps`           | int        | 最大步数限制，达到该步数后自动截断回合                                                     |
| `enemy_locations`     | list       | 敌人初始位置，给定则为确定位置[(x, y), ...]，传递[(None, None), ...]则为随机在安全区域出生 |
| `fixed_agent_loc`     | tuple/None | Agent 初始位置，(x, y)为确定位置，(None, None)则为随机在安全区域出生                       |
| `fixed_goal_loc`      | tuple/None | 目标初始位置，(x, y)为确定位置，(None, None)则为随机在安全区域出生                         |
| `danger_radius`       | int        | 危险区域影响半径，供危险区生成函数使用                                                     |
| `danger_threshold`    | float      | 危险区域的判定阈值，Agent 所处位置的危险系数高于此值则进行惩罚并终止环境                   |
| `init_safe_threshold` | float      | 出生时，安全区域的危险系数判定阈值，只有低于此值的位置才可以被选中为安全出生位置           |
| `danger_func`         | callable   | 危险区生成函数，可以根据需求进行自定义                                                     |
| `use_global_obs`      | bool       | Agent 的观测是全局的（获取地图全貌的网格信息和危险系数信息），还是局部视野                 |
| `vision_radius`       | int        | 若为局部视野，该视野的半径                                                                 |
| `render_mode`         | str        | 渲染模式("human"窗口/"rgb_array"图像)                                                      |
| `debug_mode`          | bool       | 是否输出调试信息                                                                           |

说明：

- `danger_func` 支持自定义函数，实现任何想需要的危险区分布。
- 各类 Agent 的出生位置参数支持固定与随机混合，便于多样性实验及泛化性能提升。

## 快速上手

### 安装依赖

```bash
pip install gymnasium matplotlib numpy # 基础依赖
pip install torch tensorboard # 可选依赖（PPO脚本）
```

### 环境测试

```bash
# 环境交互简单测试
python run_test.py
```

## PPO 算法示例（可选）

包含了一个借鉴 CleanRL 的 PPO 训练、测试脚本。使用 PyTorch 作为底层框架。

### 网络结构

附带的 PPO 实现采用如下网络结构：

- **输入**：
  - `grid`（离散网格，嵌入编码后输入卷积层）
  - `danger`（连续危险概率，单通道卷积）
  - `rel_goal`（相对目标坐标，直接拼接）
- **特征提取**：
  - `grid` 经过嵌入层和 2 层卷积+ReLU
  - `danger` 经过 2 层卷积+ReLU
  - 两者展平后与 `rel_goal` 拼接
- **输出**：
  - Actor 分支
  - Critic 分支

说明：

- 支持并行环境（SyncVectorEnv / AsyncVectorEnv）
- 支持模型保存和加载
- TensorBoard 记录关键指标和超参数
- 记录每次训练的环境参数（env_config）

### 训练与测试

```bash
# 训练
python ppo_cleanrl_torch.py --num-envs 8 --total-timesteps 500000
# 测试
python ppo_cleanrl_torch.py --test --load-path trained_models/xxx/model.pt --render
```

说明：

- 脚本仅在 Apple Mac Mini (M4) Python 3.12 上进行了测试，训练时选择设备为 MPS，尚未在 CUDA 设备进行测试。

## 目录结构

```
gymnasium_minigrid/         # 环境包
	core/ envs/ rendering/  # 相关模块
run_test.py                 # 环境交互与可视化
ppo_cleanrl_torch.py        # PPO算法的训练测试脚本
runs/                       # 日志与配置
trained_models/             # 训练模型
```

## TODO

- [ ] 添加YAML配置文件，支持环境参数和超参数的外部管理，统一实验管理
- [ ] 允许敌人 Agent 随时间推移而运动
- [ ] 添加更多的 RL 算法支持
- [ ] 添加 JAX 训练脚本的支持
- [ ] 对比不同框架和算法的表现
- [ ] 英文说明及注释

## 参考

- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://pytorch.org/)
