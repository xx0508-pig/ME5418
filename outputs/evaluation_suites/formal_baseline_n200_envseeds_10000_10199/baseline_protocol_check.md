# Baseline 协议兼容性核查

- 项目代码：`4980fa07ce9a1d2a215eeb9c4bc0a95880b28df4`，工作区干净。
- PPO 三 seed 套件复核：75 项、15000 回合，`per_training_seed.csv` 75 行、`aggregate_across_training_seeds.csv` 25 行，`training_seeds=[42,43,44]`；报告统计口径为独立训练 seed，未改写原始数据。

## 环境与协议一致性

| 项目 | PPO 正式评估 | baseline 脚本 | 结论 |
| --- | --- | --- | --- |
| 环境 | `ContactAwarePushing-LIBERO-v1` | `make_pushing_env("v1", ...)` | 相同 |
| 最大步数 | 300 | `max_steps=300` | 相同 |
| 动作缩放 | 0.5 | `action_scale=0.5` | 相同 |
| 控制频率 | 20 Hz | `CONTROL_FREQUENCY_HZ=20`，`control_dt=0.05` | 相同 |
| 成功条件 | 距离 ≤0.06 m、速度 ≤0.02 m/s、连续 5 步 | v1 环境同一实现 | 相同 |
| 初始状态 | 位置/yaw 随机化，yaw ±0.5 | 同一 v1 `reset(seed)` | 相同 |
| 摩擦 | 随机 0.3–0.9；固定 0.3/0.6/0.9/1.0 | 同一 `friction_range` 接口 | 相同 |
| 实际摩擦校验 | v1 `_set_surface_friction` | 同一 v1 代码 | 相同 |
| 终止原因 | stable_success/time_limit/out_of_workspace/invalid_state | 同一 `info` 字段 | 相同 |
| 接触力 | 世界坐标系净接触力 XY | `obs[9:11]`，同一 `_get_contact_force` | 相同物理量 |
| 视频 | 无 | `--video-episodes 0` | 相同 |
| 随机种子 | 10000–10199 | 将使用 10000–10199 | 相同 |

## 控制器参数（固定，不调参）

- `velocity_kp=0.20`、`velocity_ki=0.05`、`max_object_speed=0.06`、`slowdown_distance=0.14`
- 其余参数使用 `FeedbackControllerConfig` 默认值（approach/descend/push/settle 增益与限幅、积分限幅、lead 范围等）
- `pi_no_release`：`enable_settle=False`，PI 持续推动到终止。
- `hybrid_pi_settle`：`enable_settle=True`，盘子首次进入 0.06 m 后进入显式 SETTLE 阶段，后撤 0.04 m、抬升 0.08 m 释放接触；仅当盘子超过 0.075 m（0.06+0.015 回滞）才重新接近。

## 指标映射与差异

| 指标 | PPO | baseline | 是否可直接比较 |
| --- | --- | --- | --- |
| 稳定成功率 | `stable_success_rate`（每 seed 200 回合） | `success_rate`（200 回合） | 是；PPO 报三 seed 均值/样本标准差，baseline 报 200 回合成功率与 Wilson 区间 |
| 回合长度 | `mean_episode_length`（含失败） | `mean_episode_length`（含失败） | 是 |
| 最终距离 | `mean_final_goal_distance`（含失败） | `mean_final_goal_distance`（含失败） | 是 |
| episode 峰值 XY 力 | `mean_episode_peak_contact_force_xy` | `mean_max_contact_force_xy` | 是；baseline 初始化包含 reset 瞬时力，实测 reset 力约 5e-5 N，可忽略 |
| episode 超阈值率 | `contact_force_episode_exceedance_rate`（阈值 0.20） | 由 `max_contact_force_xy>0.20` 计算 | 是 |
| 步级超阈值率 | `contact_force_step_exceedance_rate` | baseline 不保存逐步力序列 | **不比较** |
| 终止原因 | `termination_counts` | `termination_counts` | 是（PPO 为三 seed 合计 600，baseline 为 200） |
| 控制 effort / 最小距离 / 进入目标时间 | 无同定义字段 | 有 | **不比较** |

## 已有 baseline 结果

- `outputs/runs/local_setup_baseline_20261001` 只有 2 回合、seed 2000、1 个视频，且是安装冒烟检查；与 200 回合/10000–10199/无视频协议不兼容，不复用。
- 未发现其他正式 feedback baseline 结果。

## 最小预检

- 1 回合/控制器（seed 2000，随机摩擦）：两者均 `time_limit`，final goal=0.0619 m，脚本与加载正常。
- 5 回合/控制器（seed 2000–2004，仅用于验证 SETTLE 路径和耗时）：`pi_no_release` 1/5，`hybrid_pi_settle` 3/5；hybrid 成功回合最终阶段为 `settle`，确认释放阶段确实触发。
- 预检不用于调参；所有正式运行使用上述固定默认参数。

