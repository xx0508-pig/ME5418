# 本地部署与学习指南

> 2026-10-13 已更新代码：训练默认归一化与正式评估协议发生变化，请先阅读 [本次更新说明](UPDATE_20261013_ZH.md)。下面的早期示例若不传 normalization，将采用新版默认值。

本项目研究：让机械臂在不知道摩擦系数的情况下，把物体推到目标并停稳。

本次做了什么、测试结果和生成文件，见 [部署操作记录](DEPLOYMENT_LOG_ZH.md)。

## 代码从哪里来

- 仓库：https://github.com/xx0508-pig/ME5418
- 部署起点：`3118365f36e2e01b3bc27b3d01edc6408dfbedd3`
- 本地完整保留 `.git`、提交历史、`origin` 和 `main` 的远程跟踪关系。
- 本次部署新增的说明和辅助脚本会显示在 `git status` 中，方便检查后自己提交。

## 先认识几个词

| 名称 | 可以怎样理解 |
| --- | --- |
| MuJoCo | 模拟物体运动、碰撞、摩擦的物理引擎 |
| LIBERO / robosuite | 提供机器人、桌面、物体与任务场景 |
| Gymnasium 环境 | 算法与仿真之间的接口：给动作，返回观测和奖励 |
| PPO | 让机器人通过反复尝试学习策略的强化学习算法 |
| PI baseline | 人工写好的反馈控制规则，用来和 PPO 公平比较 |
| episode | 从场景初始化到成功或失败的一次尝试 |
| timestep | 算法选择并执行一次动作 |
| seed | 随机数的起点，便于复现实验 |
| checkpoint | 训练中途保存的模型 |
| history k=4 | 把当前观测和最近三次动作/观测一起交给策略 |

## 建议从哪里阅读代码

1. `README.md`：组员的整体说明和实验计划。
2. `tasks/clean_tabletop_push_v1.bddl`：桌面、盘子和目标区域。
3. `envs/pushing_env_v1.py`：观测、动作、奖励、摩擦与成功条件。
4. `controllers/conventional_feedback.py`：传统控制器。
5. `envs/history_wrapper.py`：如何拼接历史信息。
6. `scripts/train_ppo.py` / `scripts/evaluate_ppo.py`：训练和评估。

## Proposal 和当前代码的关系

Proposal 描述了未知摩擦、稳定到达目标、PI 对照、PPO 以及历史信息和接触力消融。
目前仓库已实现上述实验的主要工程基础，但不要把“能运行”理解为“实验结论已经完成”。
README 说明多 seed 正式训练与比较仍是主要后续工作。

与 proposal 对照时应注意：附件写的是方块、二维速度动作；目前场景使用盘子，
环境动作空间是三维 XYZ。是否需要进一步统一实验设定，应与组员讨论后再改；
本次部署不替你们改变研究定义。

## Git 协作：本地修改怎样上传

`clone` 是下载带历史的完整项目；`commit` 是本地存档；`push` 才是上传。
安装环境和生成模型都不等于上传了代码。

开始新工作前，在没有未提交修改时：

```bash
git status
git switch main
git pull --ff-only origin main
git switch -c feature/my-change
```

`feature/my-change` 换成这次任务的名字，例如 `feature/plot-results`。
已有该分支时用 `git switch feature/my-change`，不要重复 `-c` 创建。

修改并验证后：

```bash
git status
git diff
git add 你实际修改的文件路径
git diff --cached
git commit -m "简要说明修改内容"
git push -u origin feature/my-change
```

然后在 GitHub 发起 Pull Request，请组员看过后合并。
不要直接复制含“你实际修改的文件路径”的命令，要替换为真正文件名。
不要一开始就 `git add .`，先看看有哪些文件，避免上传无关数据。
若 `pull --ff-only` 失败，先检查分支分歧；不要用 `reset --hard` 或强制推送来解决。
正在训练时不要切换分支或修改训练使用的源码。

能下载公开仓库不代表有 push 权限。上传时需要你自己的 GitHub 登录和仓库写权限；
若没有权限，请组员添加 collaborator，或使用自己的 fork 再提交 PR。
GitHub HTTPS 登录使用凭据管理器或 token，不能用账户密码作为 Git 密码。
不要把 token 写入代码、远程地址或文档。

如果第一次 commit 提示缺少姓名邮箱，只给这个仓库设置即可：

```bash
git config user.name "你的名字"
git config user.email "你的 GitHub 邮箱或 noreply 邮箱"
```

## 哪些内容适合提交

提交源码、测试、说明、依赖配置，以及有意整理的小型结果汇总。
`.venv/` 是安装好的运行环境，训练输出在 `outputs/runs/`，均应留在本地。
组员已有的 `outputs/models/*.zip` 是仓库原有模型，不代表本次训练结果。
新实验请使用独立 run name；同名输出目录会被程序拒绝覆盖。

## 日常运行：不用每次手动激活 Conda

在项目根目录打开终端。下面的 `bash scripts/local.sh` 相当于“使用这个项目专属的 Python”，
并自动设置缓存、无窗口渲染和 CPU 线程数；它不会修改你的 shell 配置。

```bash
bash scripts/local.sh --version
bash scripts/local.sh -m pytest -q
bash scripts/local.sh -m scripts.smoke_test_clean_env
```

看命令有哪些可调选项：

```bash
bash scripts/local.sh -m scripts.train_ppo --help
bash scripts/local.sh -m scripts.evaluate_ppo --help
bash scripts/local.sh -m scripts.evaluate_feedback_baseline --help
```

自己开始一次短训练（自动生成实验目录名）：

```bash
bash scripts/local.sh -m scripts.train_ppo \
  --env-version v1 --observation-mode posvel9 --history-length 1 \
  --timesteps 4096 --checkpoint-freq 2048 \
  --friction-min 0.3 --friction-max 0.9 --seed 42
```

结尾会打印保存路径。把那次的 run directory 用于评估，例如将下面的
`实际实验目录名` 替换为刚刚生成的名称：

```bash
bash scripts/local.sh -m scripts.evaluate_ppo \
  --run-dir outputs/runs/实际实验目录名 \
  --episodes 2 --video-episodes 1 --seed 2000 \
  --friction-min 0.3 --friction-max 0.9
```

这是开发检查，不是最终统计；正式评估要按 README 统一 seed、摩擦和样本量。
PPO 每批收集 2048 步，因此很小的 `--timesteps` 也可能实际执行至少 2048 步。

不训练也能运行传统控制器：

```bash
bash scripts/local.sh -m scripts.evaluate_feedback_baseline \
  --controller-variant pi_no_release \
  --episodes 2 --video-episodes 1 --seed 2000 \
  --friction-min 0.3 --friction-max 0.9
```

查看训练曲线：

```bash
bash scripts/local.sh -m tensorboard.main --logdir outputs/runs --host 127.0.0.1 --port 6006
```

在浏览器打开终端提示的本机地址；终端按 Ctrl+C 停止 TensorBoard。
仿真默认无窗口运行，画面由评估脚本保存为 MP4，而不是自动弹出机器人窗口。

## 换电脑或重建环境

本机已经做过的安装不用每天重复。在新的 Linux checkout 中，可依次运行：

```bash
conda env create -p "$PWD/.venv" -f environment.yml
bash scripts/local.sh scripts/prepare_local.py
bash scripts/local.sh -m pip check
bash scripts/local.sh -m pytest -q
```

`prepare_local.py` 下载固定版本的官方 LIBERO 资产，并按文件哈希校验完整性，
把配置写入 `.local/libero-config/config.yaml`，把包的资产目录链接到 `.local/libero-assets`。
资源约 403 MiB。重复执行会校验和复用已经下载的文件，不需要下载示范数据集。
若下载中断，再运行同一命令即可。它保留已有配置，不覆盖其他资产目录。
独立环境本身含绝对路径，不要把 `.venv` 复制给组员；分享代码和依赖配置后重建。

新电脑需要可用的 EGL / Mesa 或 NVIDIA 图形驱动。本机已有 Mesa，所以无需安装系统包。
正式训练前必须在新电脑重新验证，不能只凭依赖版本相同就认为结果一定一致。

## IDE 如何选择 Python

VS Code / PyCharm 的 Python interpreter 选择项目下面的 `.venv/bin/python`。
直接点击 IDE 的 Run 按钮不会自动加载 `scripts/local.sh` 的环境变量；
初学时建议在 IDE 终端使用上面的启动命令，排错更容易。

## 本机硬件说明

本机有 NVIDIA RTX 4060 Laptop（8 GiB 显存）与 AMD 集成显卡。
在主机权限下，NVIDIA 驱动 580.178.04 和 CUDA 13.0 正常；无需修复驱动。
本次工具的受限沙箱不能访问 GPU，因此最初看到 `nvidia-smi` 失败，
但这不是你的系统故障。CPU/Mesa 路径已通过测试，主机 GPU 路径另行验证并记入部署记录。
启动脚本检测到 NVIDIA 可用时保留正常 EGL 配置；不可访问时才使用 Mesa 软件渲染。
PPO 脚本由 Stable-Baselines3 自动选择设备。小型 MLP 不保证 GPU 训练一定更快，
但 GPU 渲染通常有帮助；正式实验应记录实际运行设备。

## 精确版本记录

- `environment.yml`：组员维护的主要依赖配置，本次保持原样。
- `docs/requirements-local-freeze.txt`：这次实际安装的全部 pip 包版本。
- `docs/conda-linux-64-explicit.txt`：这次 Conda 基础包的精确 Linux 构建。

普通使用优先按 environment.yml 建环境。需要尽量还原本次全部传递依赖时，
在**尚未存在 `.venv` 的新 checkout** 中执行：

```bash
conda create -p "$PWD/.venv" --file docs/conda-linux-64-explicit.txt
.venv/bin/python -m pip install -r docs/requirements-local-freeze.txt
bash scripts/local.sh scripts/prepare_local.py
bash scripts/local.sh -m pip check
bash scripts/local.sh -m pytest -q
```

精确清单是本次 Linux x86_64 环境快照，不适合直接用于 macOS 或 Windows。
Conda 的 explicit 文件不包含 pip 包，因此两步安装都需要执行。
