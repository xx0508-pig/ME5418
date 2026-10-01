# Contact-Aware Pushing with LIBERO

基于 LIBERO、MuJoCo 和 PPO 的接触感知桌面推盘研究项目。项目比较传统 PI
反馈控制与强化学习策略，并研究速度、接触力和短时历史信息对不同摩擦条件下
稳定推送的影响。

A contact-aware tabletop pushing project built with LIBERO, MuJoCo, and PPO.
It compares conventional PI feedback control with reinforcement learning and
studies whether velocity, contact force, and short action-observation histories
improve stable pushing across different friction conditions.

- [中文说明](#中文说明)
- [English documentation](#english-documentation)

---

## 中文说明

### 1. 研究目标

项目使用 Franka Panda 机械臂把盘子推入桌面上的目标区域。正式 benchmark
不仅要求盘子到达目标，还要求盘子真正稳定下来：

- 目标距离不超过 `0.06 m`；
- 物体速度不超过 `0.02 m/s`；
- 上述两个条件连续满足 `5` 个控制步；
- 控制频率为 `20 Hz`，因此稳定时间约为 `0.25 s`。

每个 episode 最多运行 300 步，并明确记录下列终止原因：
`stable_success`、`time_limit`、`out_of_workspace` 和 `invalid_state`。

项目保留两个环境版本：

| 环境 | 用途 |
| --- | --- |
| `ContactAwarePushing-LIBERO-v0` | 兼容并复现早期模型和结果 |
| `ContactAwarePushing-LIBERO-v1` | 正式 benchmark；使用稳定成功条件、明确失败原因和更强的初始状态随机化 |

v1 在每次 reset 时随机化盘子的初始 XY 位置、yaw 和真实接触摩擦。代码还会
读取 MuJoCo 的实际 contact friction，防止“请求值”和“仿真实际值”不一致。

### 2. 当前进展

已经完成的工程与实验基础包括：

- v0/v1 环境并存，旧模型的语义不会被新 benchmark 改写；
- 真实接触摩擦检查，以及 reset 后的运行时记录；
- 独立 run directory、`config.json`、checkpoint、TensorBoard、CSV/JSON 评估结果；
- 盘子初始位置、yaw 与摩擦的可复现随机化；
- 无释放阶段的 PI baseline 和带 `SETTLE` 释放阶段的 hybrid baseline；
- `pos6`、`posvel9`、`posvel_force11` 三种正式观测；
- `k=1` memoryless 和 `k=4` action-observation history；
- PPO actor/critic 均使用两层 `128 -> 128` ReLU 网络；
- pytest 回归测试覆盖环境、摩擦、随机种子、成功/失败边界、控制器、history
  wrapper 和脚本配置。

当前主要工作是完成 observation/history 的多 seed 正式训练与比较。模型和
生成的实验目录默认不提交到 GitHub，以免仓库膨胀；每位协作者只需共享配置、
汇总 CSV/JSON 和必要的最终模型。

### 3. 项目结构

```text
contact_aware_pushing/
├── controllers/
│   └── conventional_feedback.py   # PI 与 hybrid PI+SETTLE 控制器
├── envs/
│   ├── custom_libero_env.py       # 项目本地 LIBERO 环境适配层
│   ├── pushing_env.py             # 兼容旧实验的 v0 环境
│   ├── pushing_env_v1.py          # 正式 benchmark v1
│   ├── observation_modes.py       # 观测模式及维度定义
│   └── history_wrapper.py         # 固定窗口动作—观测历史 wrapper
├── scripts/
│   ├── train_ppo.py               # PPO 训练与可复现实验记录
│   ├── evaluate_ppo.py            # PPO 批量评估、CSV/JSON 与视频输出
│   ├── evaluate_feedback_baseline.py
│   └── smoke_test_clean_env.py
├── tasks/
│   ├── clean_tabletop_push.bddl   # v0 LIBERO 任务
│   └── clean_tabletop_push_v1.bddl# v1 起点/目标区域定义
├── tests/                          # pytest 回归测试
├── outputs/                        # 模型和实验输出；多数生成物被 Git 忽略
├── environment.yml                 # 推荐的 Conda/Python 环境
└── pytest.ini
```

BDDL 是 LIBERO 用来描述任务对象、初始区域和目标条件的任务定义语言。它描述
“场景是什么”，而 Gymnasium 环境代码负责观测、动作、奖励、随机化和终止逻辑。

### 4. 安装与复现环境

推荐 Ubuntu/Linux、Conda 和支持 EGL 的 NVIDIA 驱动。先克隆项目，再由
`environment.yml` 创建独立环境：

```bash
git clone https://github.com/xx0508-pig/ME5418.git
cd ME5418
conda env create -f environment.yml
conda activate libero-rl
```

`environment.yml` 固定了当前已经验证的 Python、LIBERO、MuJoCo、Gymnasium、
Stable-Baselines3 和 PyTorch 版本，也把 LeRobot 固定到一个明确的 Git commit。
第一次运行时 LIBERO 可能会自动下载资产。若组员的 CUDA/显卡平台不同，可以先
创建此环境；如 PyTorch 安装或 GPU 识别失败，再根据该机器的 CUDA 版本单独安装
兼容的 PyTorch，项目本身无需修改。

检查环境：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MUJOCO_GL=egl \
python -m pytest -q
```

这里禁用系统中无关的 pytest 插件，并让 MuJoCo 使用无窗口 EGL 渲染。测试全部
通过后，再开始长时间训练。

### 5. 观测模式与 history

正式实验使用以下三种单步观测：

| 模式 | 内容 | 单步维度 |
| --- | --- | ---: |
| `pos6` | 末端执行器—物体、物体—目标的相对位置 | 6 |
| `posvel9` | `pos6` + 物体 XYZ 速度 | 9 |
| `posvel_force11` | `posvel9` + 接触力 XY | 11 |

`posforce9` 和 `posvelforce12` 仅用于复现旧模型。正式力观测不使用 Z 方向，
因为 Z 分量容易被盘子重力和桌面支撑力主导。

`--history-length 1` 表示无显式历史，策略只接收当前观测。`--history-length 4`
表示把当前状态和最近三组动作/观测拼接起来：

```text
[o_t, a_(t-1), o_(t-1), a_(t-2), o_(t-2), a_(t-3), o_(t-3)]
```

因此，`posvel9 + k=4` 是 45D，`posvel_force11 + k=4` 是 53D。reset
时四个 observation slot 都填初始观测，三个 past-action slot 填零。这样普通
feed-forward PPO 也能根据短期变化推断运动趋势和接触过程，不需要更换成 RNN。

### 6. 快速开发训练

长训练前先运行 4,096 步，检查目录、配置和 checkpoint 能否生成：

```bash
MUJOCO_GL=egl python -m scripts.train_ppo \
  --env-version v1 \
  --observation-mode posvel9 \
  --history-length 1 \
  --timesteps 4096 \
  --run-name dev_smoke_posvel9_k1_seed42 \
  --save-name ppo_v1 \
  --friction-min 0.3 \
  --friction-max 0.9 \
  --checkpoint-freq 2048 \
  --seed 42
```

每次训练写入独立目录：

```text
outputs/runs/<run_name>/
├── config.json
├── final_model.zip
├── checkpoints/
├── tensorboard/
└── evaluations/
```

不要重复使用已有的 `run_name`，脚本会拒绝覆盖旧实验。

### 7. 正式 PPO 训练

下面是一个 500k 步、v1、`posvel9 + k=1` 的例子：

```bash
MUJOCO_GL=egl python -m scripts.train_ppo \
  --env-version v1 \
  --observation-mode posvel9 \
  --history-length 1 \
  --timesteps 500000 \
  --run-name ppo_v1_posvel9_k1_seed42 \
  --save-name ppo_v1 \
  --friction-min 0.3 \
  --friction-max 0.9 \
  --checkpoint-freq 50000 \
  --seed 42
```

建议正式消融使用相同超参数，只改变 observation/history 和 seed。基础矩阵为：

| 实验 | 观测 | History |
| --- | --- | ---: |
| A | `pos6` | 1 |
| B | `posvel9` | 1 |
| C | `posvel9` | 4 |
| D | `posvel_force11` | 1 |
| E | `posvel_force11` | 4 |

建议至少使用 seed `42`、`43`、`44`。开发校准 seed 与最终测试 seed 必须分开，
例如开发使用 `2000–2019`，最终测试使用 `4000–4099`。

### 8. PPO 评估

使用 `--run-dir` 时，评估脚本会从训练生成的 `config.json` 自动恢复环境版本、
观测模式和 history 长度：

```bash
MUJOCO_GL=egl python -m scripts.evaluate_ppo \
  --run-dir outputs/runs/ppo_v1_posvel9_k1_seed42 \
  --evaluation-name final_seeds_4000_4099 \
  --episodes 100 \
  --video-episodes 3 \
  --seed 4000 \
  --friction-min 0.3 \
  --friction-max 0.9
```

输出包括逐回合 CSV、summary JSON 和可选 MP4。JSON 会明确记录训练模型路径、
训练/评估环境版本、seed、摩擦设置和终止原因，便于后续统计和审计。

### 9. Conventional feedback baseline

原始 PI 控制器不包含人工释放阶段：

```bash
MUJOCO_GL=egl python -m scripts.evaluate_feedback_baseline \
  --controller-variant pi_no_release \
  --run-name feedback_pi_no_release_seed2000 \
  --episodes 20 \
  --video-episodes 3 \
  --seed 2000 \
  --friction-min 0.3 \
  --friction-max 0.9
```

带 `SETTLE` 的 hybrid 版本用于分析显式接触释放是否有帮助：

```bash
MUJOCO_GL=egl python -m scripts.evaluate_feedback_baseline \
  --controller-variant hybrid_pi_settle \
  --run-name feedback_hybrid_settle_seed2000 \
  --episodes 20 \
  --video-episodes 3 \
  --seed 2000 \
  --friction-min 0.3 \
  --friction-max 0.9
```

两种方案都保留并单独报告，避免把专门为稳定条件设计的 `SETTLE` 阶段隐藏在
“传统 PI baseline”中。控制器不读取摩擦或接触力，接触力只作为评估指标。

### 10. 协作与 Git 约定

- `main` 保存经过测试、可复现的代码；短期功能可在临时 feature branch 开发；
- 不提交 `outputs/runs/`、`outputs/calibration/`、TensorBoard、视频或 checkpoints；
- 提交代码时同时提交与该代码匹配的测试和说明；
- 分享训练任务时，先同步相同 commit，再给每位组员分配不同的实验名和 seed；
- 训练结束后共享 `config.json`、评估 CSV/JSON、Git SHA 和必要的最终模型；
- 正在运行的训练已把代码和模型载入内存，正常的 `git add/commit/push` 不会使它
  停止；但训练期间不要切换分支、重置或修改训练所使用的源码。

常用安全流程：

```bash
git status
git diff
git add <修改过的文件>
git diff --cached
git commit -m "简短说明这次改了什么"
git push origin main
```

---

## English documentation

### 1. Objective and benchmark

This project trains a Franka Panda robot to push a plate into a target region.
The formal v1 benchmark defines success as both accurate placement and physical
settling:

- goal distance `<= 0.06 m`;
- object speed `<= 0.02 m/s`;
- both conditions hold for 5 consecutive control steps;
- the controller runs at 20 Hz, so the stable interval is about 0.25 seconds.

An episode lasts at most 300 steps. Its terminal reason is recorded as
`stable_success`, `time_limit`, `out_of_workspace`, or `invalid_state`.

Two environment versions are intentionally retained. v0 reproduces legacy
models and results; v1 is the formal benchmark with stable success, explicit
failure reasons, and randomized initial position, yaw, and runtime contact
friction. The environment verifies MuJoCo's active contact friction after reset
instead of trusting only the requested value.

### 2. Implemented components

- Backward-compatible v0 and formal v1 Gymnasium environments.
- Reproducible run directories with config, checkpoints, TensorBoard, and
  CSV/JSON evaluation artifacts.
- Runtime friction validation and seeded position/yaw/friction randomization.
- Conventional PI and hybrid PI-with-SETTLE baselines.
- Three formal observation modes and fixed action-observation history stacking.
- Separate two-layer `128 -> 128` ReLU actor and critic networks.
- Regression tests for the environments, randomization, termination boundaries,
  feedback controllers, history wrapper, and script configuration.

The current experimental stage is the multi-seed observation/history ablation.

### 3. Installation

Ubuntu/Linux, Conda, and an EGL-capable NVIDIA setup are recommended:

```bash
git clone https://github.com/xx0508-pig/ME5418.git
cd ME5418
conda env create -f environment.yml
conda activate libero-rl
```

The environment pins the versions verified by the project, including the exact
LeRobot source revision. LIBERO may download its assets on the first launch. If
a collaborator uses a different CUDA platform, create the environment first and
then reinstall the matching PyTorch build if GPU detection fails.

Run the regression suite before starting an experiment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MUJOCO_GL=egl \
python -m pytest -q
```

### 4. Observation and memory settings

| Mode | Single-step content | Dimension |
| --- | --- | ---: |
| `pos6` | End-effector/object and object/goal relative positions | 6 |
| `posvel9` | `pos6` plus object XYZ velocity | 9 |
| `posvel_force11` | `posvel9` plus XY contact force | 11 |

`--history-length 1` is memoryless. With `--history-length 4`, the wrapper
provides `[o_t, a_(t-1), o_(t-1), ..., a_(t-3), o_(t-3)]`. This produces 45
dimensions for `posvel9` and 53 for `posvel_force11`. At reset, observation
slots repeat the initial observation and previous-action slots contain zeros.

### 5. Training

A short reproducibility smoke run:

```bash
MUJOCO_GL=egl python -m scripts.train_ppo \
  --env-version v1 \
  --observation-mode posvel9 \
  --history-length 1 \
  --timesteps 4096 \
  --run-name dev_smoke_posvel9_k1_seed42 \
  --save-name ppo_v1 \
  --friction-min 0.3 \
  --friction-max 0.9 \
  --checkpoint-freq 2048 \
  --seed 42
```

A formal 500k-step run uses the same command with a unique run name,
`--timesteps 500000`, and normally `--checkpoint-freq 50000`. Keep all PPO
settings fixed when comparing observation modes and history lengths. Use at
least seeds 42, 43, and 44 for each matrix cell.

Every run is written to `outputs/runs/<run_name>/` and contains `config.json`,
`final_model.zip`, `checkpoints/`, `tensorboard/`, and `evaluations/`. Existing
run directories are never overwritten.

### 6. Evaluation

```bash
MUJOCO_GL=egl python -m scripts.evaluate_ppo \
  --run-dir outputs/runs/ppo_v1_posvel9_k1_seed42 \
  --evaluation-name final_seeds_4000_4099 \
  --episodes 100 \
  --video-episodes 3 \
  --seed 4000 \
  --friction-min 0.3 \
  --friction-max 0.9
```

The evaluator reconstructs the observation and history pipeline from the run
configuration. It exports per-episode CSV data, a JSON summary, and optional
videos. Use separate development and final seeds; the recommended ranges are
2000–2019 and 4000–4099, respectively.

### 7. Feedback baselines

Evaluate the original PI controller with `--controller-variant pi_no_release`.
Use `--controller-variant hybrid_pi_settle` for the explicitly released hybrid
variant. Both variants are reported separately because the SETTLE phase is a
task-specific design choice rather than an implicit advantage of conventional
feedback control. Full commands are provided in the Chinese section above and
are also available through:

```bash
python -m scripts.evaluate_feedback_baseline --help
```

### 8. Collaboration and reproducibility

Keep tested code on `main` and use short-lived feature branches only while a
change is under development. Generated runs, checkpoints, videos, and local
calibration outputs are intentionally ignored by Git. Collaborators should
start from the same Git commit, use unique run names and assigned seeds, and
return the run configuration, evaluation CSV/JSON, Git SHA, and only the final
models that are needed for comparison.

Committing or pushing documentation while a training process is running does
not stop that process. Avoid changing branches, resetting the repository, or
editing source files used by the active training job until it finishes.
