# ME5418 传统控制器 baseline 与三 seed PPO 正式对比

- 完成时间：2026-11-17 16:25:08（Asia/Singapore）
- 基线套件：`formal_baseline_n200_envseeds_10000_10199`；2 种控制器 × 5 种摩擦条件 × 200 回合 = 10 项、2000 回合。
- PPO 套件：`formal_multiseed_n200_envseeds_10000_10199`；5 种配置 × 5 种条件 × 3 个训练 seed × 200 回合 = 75 项、15000 回合。
- 本次不调参、不重训 PPO、不改课程报告、不 push；PPO 三 seed 数据哈希未变。

## 结论摘要

- 五条件平均稳定成功率：PI no release **24.8%**，Hybrid PI+SETTLE **55.8%**，PPO 五种配置 **84.8%–96.9%**。PPO 在全部 25 个“配置×条件”单元中都高于两种 baseline。
- 但 baseline 的净接触力 XY 代理指标明显更低：五条件平均 episode 峰值 **0.045**，2000 回合中没有 episode 超过 0.20；PPO 五条件平均峰值为 **0.102–0.161**，episode 超阈值率 **10.6%–28.0%**。这是成功率与接触力代理指标之间的取舍，不是完整安全证明。
- Hybrid 相对 PI 的主要收益是把“进入目标区”转化为“稳定成功”：两种控制器在每个条件的 `ever_distance_success` 相同，但 Hybrid 的稳定成功率高得多；成功回合最终停在 SETTLE 阶段。
- 因此，数据支持“PPO 在本协议下稳定成功率更高”，但不支持“RL 必然更好”或“加入力/历史必然更好”；也不支持用净接触力 XY 单独证明安全性。

## 协议与可比性

| 项目 | PPO 正式评估 | baseline 正式评估 | 结论 |
| --- | --- | --- | --- |
| 环境 | `ContactAwarePushing-LIBERO-v1` | 同一 v1 | 相同 |
| 步数/频率 | 300 步、20 Hz | 300 步、20 Hz（`control_dt=0.05`） | 相同 |
| 动作缩放 | 0.5 | 0.5 | 相同 |
| 动作空间与限幅 | [-1,1]，×0.5 | 环境 [-1,1]，×0.5；控制器内部限幅 0.45–0.70 | 环境相同；控制器额外限幅 |
| 实际摩擦 | v1 运行时校验 | 同一 v1 代码 | 相同 |
| 成功条件 | 距离 ≤0.06 m、速度 ≤0.02 m/s、连续 5 步 | 同一 v1 实现 | 相同 |
| 初始状态 | 位置/yaw 随机化，yaw ±0.5 | 同一 `reset(seed)` | 相同 |
| 摩擦 | 随机 0.3–0.9；固定 0.3/0.6/0.9/1.0 | 同一 v1 摩擦接口 | 相同 |
| 环境 seeds | 10000–10199 | 10000–10199 | 相同 |
| 视频 | 无 | `--video-episodes 0` | 相同 |
| 接触力 | 世界坐标系净接触力 XY | 同一 `_get_contact_force`，`obs[9:11]` | 同一物理量 |

控制器参数固定为仓库已确定默认值：`velocity_kp=0.20`、`velocity_ki=0.05`、`max_object_speed=0.06`、`slowdown_distance=0.14`；其余为 `FeedbackControllerConfig` 默认值。未在测试 seeds 上调参。

**指标定义差异**：baseline 的 `mean_max_contact_force_xy` 与 PPO 的 `mean_episode_peak_contact_force_xy` 都是“每回合峰值 XY 力的回合均值”；baseline 初始化包含 reset 瞬时力，实测 reset 力约 5×10⁻⁵ N，可忽略。baseline 不保存逐步力序列，因此**不比较步级超阈值率**；episode 超阈值率按 `max_contact_force_xy>0.20` 从 baseline CSV 计算。控制 effort、最小距离、首次进入目标时间没有 PPO 同定义字段，不拼接。

## 稳定成功率对比

Baseline 为 200 回合成功率 [95% Wilson 区间]；PPO 为三个独立训练 seed 的均值 ± 样本标准差（n−1，n=3）。两者不确定性的含义不同。

| 策略 | 随机 0.3–0.9 | 固定 0.3 | 固定 0.6 | 固定 0.9 | 固定 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PI no release | 21.5% [16.4, 27.7] | 3.0% [1.4, 6.4] | 27.0% [21.3, 33.5] | 35.0% [28.7, 41.8] | 37.5% [31.1, 44.4] |
| Hybrid PI+SETTLE | 60.5% [53.6, 67.0] | 59.5% [52.6, 66.1] | 62.5% [55.6, 68.9] | 50.5% [43.6, 57.4] | 46.0% [39.2, 52.9] |
| PPO pos6_k1 | 95.0 ± 5.3 | 96.0 ± 3.8 | 95.7 ± 5.1 | 91.5 ± 12.2 | 83.2 ± 20.6 |
| PPO posvel9_k1 | 99.5 ± 0.5 | 100.0 ± 0.0 | 100.0 ± 0.0 | 90.0 ± 11.4 | 84.7 ± 17.0 |
| PPO posvel9_k4 | 89.0 ± 10.8 | 97.3 ± 4.6 | 87.8 ± 14.8 | 80.7 ± 8.7 | 83.5 ± 7.8 |
| PPO force11_k1 | 99.5 ± 0.0 | 100.0 ± 0.0 | 99.8 ± 0.3 | 95.7 ± 5.4 | 89.5 ± 5.2 |
| PPO force11_k4 | 90.3 ± 10.3 | 99.0 ± 1.0 | 89.7 ± 14.1 | 76.8 ± 7.5 | 68.3 ± 26.5 |

## Baseline 逐条件明细

回合长度与最终距离均包含失败回合。`ever_distance` 是曾经进入 0.06 m 的回合比例（仅 baseline 记录）。

| 控制器 | 条件 | 稳定成功 [95% CI] | 平均长度 | 最终距离 | 峰值 XY | episode >0.20 | time_limit | ever_distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PI no release | 随机 0.3–0.9 | 21.5% [16.4, 27.7] | 296.3 | 55.6 mm | 0.043 | 0.0% | 78.5% | 73.5% |
| PI no release | 固定 0.3 | 3.0% [1.4, 6.4] | 299.8 | 50.7 mm | 0.055 | 0.0% | 97.0% | 84.0% |
| PI no release | 固定 0.6 | 27.0% [21.3, 33.5] | 295.5 | 55.6 mm | 0.042 | 0.0% | 73.0% | 72.0% |
| PI no release | 固定 0.9 | 35.0% [28.7, 41.8] | 294.1 | 69.9 mm | 0.042 | 0.0% | 65.0% | 58.5% |
| PI no release | 固定 1.0 | 37.5% [31.1, 44.4] | 294.1 | 74.4 mm | 0.043 | 0.0% | 62.5% | 53.5% |
| Hybrid PI+SETTLE | 随机 0.3–0.9 | 60.5% [53.6, 67.0] | 287.8 | 65.8 mm | 0.043 | 0.0% | 39.5% | 73.5% |
| Hybrid PI+SETTLE | 固定 0.3 | 59.5% [52.6, 66.1] | 287.0 | 63.5 mm | 0.055 | 0.0% | 40.5% | 84.0% |
| Hybrid PI+SETTLE | 固定 0.6 | 62.5% [55.6, 68.9] | 288.1 | 65.1 mm | 0.042 | 0.0% | 37.5% | 72.0% |
| Hybrid PI+SETTLE | 固定 0.9 | 50.5% [43.6, 57.4] | 291.6 | 74.6 mm | 0.042 | 0.0% | 49.5% | 58.5% |
| Hybrid PI+SETTLE | 固定 1.0 | 46.0% [39.2, 52.9] | 292.4 | 78.0 mm | 0.043 | 0.0% | 54.0% | 53.5% |

补充：baseline 全部 2000 回合的全局最大 episode 峰值力为 0.183（随机条件），仍低于 0.20；在相同摩擦条件下 PI 与 Hybrid 的全局峰值力相同，说明 SETTLE 没有引入更高峰值力。

## 五条件等权平均（描述性汇总）

| 策略 | 稳定成功 | 平均长度 | 最终距离 | 峰值 XY | episode >0.20 | time_limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PI no release | **24.8%** | 296.0 | 61.3 mm | 0.045 | 0.0% | 75.2% |
| Hybrid PI+SETTLE | **55.8%** | 289.4 | 69.4 mm | 0.045 | 0.0% | 44.2% |
| PPO pos6_k1 | **92.3%** | 114.1 | 28.7 mm | 0.102 | 10.6% | 7.7% |
| PPO posvel9_k1 | **94.8%** | 97.8 | 14.1 mm | 0.117 | 18.0% | 5.2% |
| PPO posvel9_k4 | **87.7%** | 116.0 | 40.5 mm | 0.160 | 26.4% | 12.3% |
| PPO force11_k1 | **96.9%** | 93.8 | 13.7 mm | 0.122 | 15.2% | 3.1% |
| PPO force11_k4 | **84.8%** | 130.5 | 43.9 mm | 0.161 | 28.0% | 15.2% |

## 固定摩擦 1.0（高摩擦泛化）

| 策略 | 稳定成功 | 平均长度 | 最终距离 | 峰值 XY | episode >0.20 | time_limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PI no release | 37.5% [31.1, 44.4] | 294.1 | 74.4 mm | 0.043 | 0.0% | 62.5% |
| Hybrid PI+SETTLE | 46.0% [39.2, 52.9] | 292.4 | 78.0 mm | 0.043 | 0.0% | 54.0% |
| PPO pos6_k1 | 83.2% | 143.8 | 41.8 mm | 0.119 | 12.8% | 16.8% |
| PPO posvel9_k1 | 84.7% | 126.2 | 31.1 mm | 0.155 | 27.8% | 15.3% |
| PPO posvel9_k4 | 83.5% | 129.7 | 43.5 mm | 0.199 | 38.8% | 16.5% |
| PPO force11_k1 | 89.5% | 114.3 | 25.6 mm | 0.154 | 21.3% | 10.5% |
| PPO force11_k4 | 68.3% | 169.0 | 61.1 mm | 0.189 | 33.8% | 31.7% |

## 失败原因

- baseline 全部 2000 回合中：`stable_success` 806（40.3%），`time_limit` 1194（59.7%）；`out_of_workspace` 0，`invalid_state` 0。
- 其中 PI 248/1000 成功、752 次 time_limit；Hybrid 558/1000 成功、442 次 time_limit。
- PPO 三 seed 合计 15000 回合：`stable_success` 13695，`time_limit` 1305，`out_of_workspace` 0，`invalid_state` 0。
- 因而两种方法的失败都只是 300 步内未达到稳定成功条件；没有越界或数值非法终止。baseline 的 `ever_distance_success` 与 PI/Hybrid 相同（如固定 0.3 都是 84%，随机都是 73.5%），说明差距主要不在“能否进入目标区”，而在“进入后能否稳定下来”。

## PI 与 Hybrid PI+SETTLE 的区别

- `pi_no_release`：接近—下降—PI 速度反馈推动，直到稳定成功或 300 步上限；没有显式释放阶段。
- `hybrid_pi_settle`：前三个阶段与 PI 相同；盘子首次进入 0.06 m 后进入 **SETTLE**，沿进入方向后撤 0.04 m、抬升 0.08 m 释放接触，让摩擦把盘子停住；只有盘子离开 0.075 m（0.06+0.015 回滞）才重新接近。
- 2000 回合中 PI 的最终阶段全部是 `push`；Hybrid 的最终阶段在成功回合中为 `settle`，说明 SETTLE 确实触发，并非只改标签。
- 因此，Hybrid 的优势是**显式设计的释放阶段**，不是“传统 PI 本身”自动具备的能力；报告必须把两者分开。

## 数据支持的发现与待验证解释

**数据支持：**
- PPO 五种配置在全部 25 个配置×条件单元中的稳定成功率都高于 PI 和 Hybrid；在固定 0.3 差距尤其大（PI 3.0%，Hybrid 59.5%，PPO 96.0–100.0%）。
- Hybrid 在全部五个条件下稳定成功率高于 PI（60.5>21.5、59.5>3.0、62.5>27.0、50.5>35.0、46.0>37.5）。
- baseline 的 episode 峰值净 XY 力均值约 0.045，2000 回合无 episode 超过 0.20；PPO 的对应均值为 0.102–0.161，episode 超阈值率 10.6%–28.0%。
- PPO 的高摩擦 seed 差异很大（例如 `posvel_force11_k4` 固定 1.0 为 77.5/38.5/89.0%），说明不能只看单 seed。
- PI 在固定 0.3 的最终距离均值反而低于 Hybrid（50.7 mm vs 63.5 mm），但稳定成功率只有 3.0% vs 59.5%；最终距离均值不能替代稳定成功指标。

**待验证解释（不是本次数据直接证明）：**
- PI 在低摩擦下稳定成功率极低，可能与盘子进入目标后仍持续滑动、难以连续 5 步满足速度阈值有关；`ever_distance` 很高而稳定成功率很低支持这一解释，但本次没有做“残余速度—失败”的受控消融。
- PPO 成功率更高与峰值力更高同时出现，只能说明本协议下的相关性，不能证明“必须用更大力才能成功”，也不能证明净 XY 力是完整安全指标。
- Hybrid 仍低于 PPO，可能是因为 SETTLE 只释放接触、不主动修正位置；这一解释需要进一步实验。
- 只评估了一组已确定控制器参数；不能据此断言“调参后的 PI 一定更差/更好”。

## 统计口径与限制

- Baseline 没有训练 seed；每个条件 200 回合，Wilson 区间只反映**回合采样不确定性**。
- PPO 的均值/样本标准差在 3 个独立训练 seed 上计算；与 baseline 的 200 回合区间不是同一种不确定性，不能直接比较区间宽窄。
- 五条件等权平均只是本实验的描述性汇总，不代表真实摩擦分布下的期望性能。
- 接触力是盘子受到的世界坐标系净接触力 XY 幅值，是代理指标，不是完整安全证明；未比较步级超阈值率（baseline 无逐步序列）。
- 回合长度与最终距离包含失败回合；成功率才是稳定成功的主指标。

## 产物与复现

- 对比报告：`outputs/evaluation_suites/formal_baseline_n200_envseeds_10000_10199/README_BASELINE_VS_PPO_ZH.md`（本文件）
- 共同指标 CSV（35 行 = 7 策略×5 条件）：`outputs/evaluation_suites/formal_baseline_n200_envseeds_10000_10199/comparison_metrics.csv`
- 成功率对比图：`outputs/evaluation_suites/formal_baseline_n200_envseeds_10000_10199/friction_success_baseline_vs_ppo.png`
- 成功率—接触力取舍图：`outputs/evaluation_suites/formal_baseline_n200_envseeds_10000_10199/success_vs_peak_force.png`
- 基线运行目录：`outputs/runs/formal_baseline_n200_envseeds_10000_10199/`
- 基线驱动脚本：`outputs/evaluation_suites/formal_baseline_n200_envseeds_10000_10199/run_formal_baseline.sh`；总日志：`outputs/evaluation_suites/formal_baseline_n200_envseeds_10000_10199/run.log`；逐任务日志：`outputs/evaluation_suites/formal_baseline_n200_envseeds_10000_10199/logs/`
- 基线清单：`artifact_manifest.json`；完成记录：`baseline_completion.json`；核验：`baseline_audit.json`；协议核查：`baseline_protocol_check.md`
- 启动记录：`launch_info.json`；PPO 保护哈希：`ppo_reference_hashes.json`
- 复现单个 baseline 条件的命令模板：
```bash
bash scripts/local.sh -m scripts.evaluate_feedback_baseline \
  --controller-variant pi_no_release \
  --output-root outputs/runs/formal_baseline_n200_envseeds_10000_10199 \
  --run-name pi_no_release_fixed_mu_0p6 \
  --episodes 200 --video-episodes 0 --seed 10000 --friction 0.6 \
  --velocity-kp 0.20 --velocity-ki 0.05 --max-object-speed 0.06 --slowdown-distance 0.14
```

## 声明

- 本次只做 baseline 正式评估与对比；未重训 PPO、未调控制器参数、未运行新 baseline 之外的实验、未修改课程提交报告、未 push。
- PPO 三 seed 原始数据与旧两 seed 快照未被改写；baseline 结果存放在独立目录。
