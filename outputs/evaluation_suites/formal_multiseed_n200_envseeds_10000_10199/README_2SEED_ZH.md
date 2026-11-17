# ME5418 seed43/44 正式评估结果（两 seed，非最终三 seed 结论）

- 生成时间：2026-11-15 23:25（Asia/Singapore）
- Suite：`formal_multiseed_n200_envseeds_10000_10199`
- 范围：**仅 seed43、seed44 两个训练 seed**；seed42 尚未收齐，本文任何数字都不是三 seed 最终结论。
- 协议：每模型/条件 200 回合，环境 seeds 10000–10199，随机摩擦 0.3–0.9 + 固定 0.3/0.6/0.9/1.0，确定性策略，无视频。
- 完成度：50/50 任务，10000/10000 回合；50 个逐回合 CSV 各 200 行；无缺失文件、无非有限数值、无视频文件。

## 版本与来源

- 评估代码提交：`4980fa07ce9a1d2a215eeb9c4bc0a95880b28df4`，工作区干净（`dirty=false`）。
- 训练基准：`0bbabb3846720343a0e4b7c3bc15fa5d429d307b`（十个模型的 `config.json` 均记录该 SHA，且训练时 `dirty=true`）。
- 评估相关代码（`scripts/evaluate_ppo.py`、`scripts/evaluate_formal_multiseed_suite.py`、`envs/`、`tests/`）相对 `0bbabb3` 无差异；`4980fa0` 只新增文档、部署辅助脚本和示例，不改变评估逻辑。
- 训练源码快照：`outputs/training_suites/seeds43_44_queue/source_20261013_221221.tar.gz`。
- 逐模型 SHA256 与来源记录：`launch_info.json`、`artifact_manifest.json`。

## 成功率（stable success）

每格为 `seed43 / seed44 / 合并 400 回合`。

| 配置 | 随机 0.3–0.9 | 固定 0.3 | 固定 0.6 | 固定 0.9 | 固定 1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pos6_k1` | 100.0% / 89.5% / **94.8%** | 100.0% / 92.5% / **96.2%** | 100.0% / 90.0% / **95.0%** | 100.0% / 77.5% / **88.8%** | 99.5% / 60.0% / **79.8%** |
| `posvel9_k1` | 99.0% / 100.0% / **99.5%** | 100.0% / 100.0% / **100.0%** | 100.0% / 100.0% / **100.0%** | 95.0% / 98.0% / **96.5%** | 94.0% / 95.0% / **94.5%** |
| `posvel9_k4` | 90.5% / 99.0% / **94.8%** | 100.0% / 100.0% / **100.0%** | 94.0% / 98.5% / **96.2%** | 74.0% / 90.5% / **82.2%** | 78.5% / 92.5% / **85.5%** |
| `posvel_force11_k1` | 99.5% / 99.5% / **99.5%** | 100.0% / 100.0% / **100.0%** | 99.5% / 100.0% / **99.8%** | 98.0% / 89.5% / **93.8%** | 95.5% / 86.5% / **91.0%** |
| `posvel_force11_k4` | 99.0% / 93.0% / **96.0%** | 98.0% / 100.0% / **99.0%** | 99.5% / 96.0% / **97.8%** | 85.5% / 72.5% / **79.0%** | 38.5% / 89.0% / **63.8%** |

## 稳定性与接触力（两 seed 平均，400 回合/格）

| 配置 | 条件 | 平均步数 | 平均最终距离 (mm, 含失败) | 平均 episode 峰值 XY 力 | episode 超阈值率 | 步级超阈值率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pos6_k1` | 随机 0.3–0.9 | 100.7 | 24.79 | 0.102 | 9.2% | 0.10% |
| `pos6_k1` | 固定 0.3 | 93.2 | 22.41 | 0.090 | 9.5% | 0.11% |
| `pos6_k1` | 固定 0.6 | 100.2 | 25.18 | 0.107 | 10.2% | 0.11% |
| `pos6_k1` | 固定 0.9 | 119.1 | 27.92 | 0.120 | 12.2% | 0.12% |
| `pos6_k1` | 固定 1.0 | 141.2 | 42.39 | 0.131 | 14.3% | 0.13% |
| `posvel9_k1` | 随机 0.3–0.9 | 85.5 | 5.00 | 0.081 | 7.5% | 0.09% |
| `posvel9_k1` | 固定 0.3 | 81.9 | 4.81 | 0.070 | 8.0% | 0.10% |
| `posvel9_k1` | 固定 0.6 | 84.6 | 4.06 | 0.079 | 5.8% | 0.07% |
| `posvel9_k1` | 固定 0.9 | 95.9 | 10.40 | 0.102 | 13.0% | 0.16% |
| `posvel9_k1` | 固定 1.0 | 102.7 | 15.24 | 0.111 | 13.0% | 0.13% |
| `posvel9_k4` | 随机 0.3–0.9 | 99.9 | 24.93 | 0.147 | 22.8% | 0.29% |
| `posvel9_k4` | 固定 0.3 | 79.2 | 8.63 | 0.117 | 14.8% | 0.21% |
| `posvel9_k4` | 固定 0.6 | 95.5 | 20.81 | 0.156 | 22.0% | 0.26% |
| `posvel9_k4` | 固定 0.9 | 136.4 | 49.44 | 0.180 | 32.5% | 0.32% |
| `posvel9_k4` | 固定 1.0 | 127.3 | 40.04 | 0.204 | 40.5% | 0.40% |
| `posvel_force11_k1` | 随机 0.3–0.9 | 86.5 | 7.37 | 0.118 | 14.2% | 0.17% |
| `posvel_force11_k1` | 固定 0.3 | 82.6 | 4.86 | 0.093 | 10.8% | 0.13% |
| `posvel_force11_k1` | 固定 0.6 | 85.4 | 6.37 | 0.123 | 17.0% | 0.22% |
| `posvel_force11_k1` | 固定 0.9 | 102.9 | 17.34 | 0.148 | 20.2% | 0.22% |
| `posvel_force11_k1` | 固定 1.0 | 110.3 | 20.87 | 0.164 | 24.0% | 0.26% |
| `posvel_force11_k4` | 随机 0.3–0.9 | 112.6 | 32.73 | 0.168 | 31.0% | 0.35% |
| `posvel_force11_k4` | 固定 0.3 | 96.2 | 26.04 | 0.139 | 22.5% | 0.28% |
| `posvel_force11_k4` | 固定 0.6 | 107.2 | 31.60 | 0.173 | 31.0% | 0.36% |
| `posvel_force11_k4` | 固定 0.9 | 160.4 | 59.65 | 0.199 | 42.5% | 0.37% |
| `posvel_force11_k4` | 固定 1.0 | 183.4 | 68.02 | 0.206 | 38.2% | 0.32% |

> 接触力是盘子受到的所有接触在世界坐标系下的净力 XY 幅值，是安全代理指标，不等同于机器人—物体之间的孤立接触力。阈值 0.20。

## 终止原因（全部 10000 回合）

- `stable_success`：9293
- `time_limit`：707
- `out_of_workspace`：0
- `invalid_state`：0

任务级没有失败；707 个 episode 失败全部是 300 步内未达到稳定成功条件。

## 需要特别说明的结果

- `posvel9_k1` 最稳定：五条件合并成功率 98.1%，固定 1.0 下两 seed 分别为 94.0%/95.0%。
- `posvel_force11_k1` 次之（96.8%），`pos6_k1` 为 90.9%，`posvel9_k4` 为 91.7%，`posvel_force11_k4` 为 87.1%。
- 高摩擦下退化明显：固定 1.0 时 `pos6_k1` 两 seed 为 99.5%/60.0%，`posvel_force11_k4` 为 38.5%/89.0%。
- **最大 seed 间差异**：`posvel_force11_k4` 固定 1.0（38.5% vs 89.0%，差 50.5 个百分点）；`pos6_k1` 固定 1.0（99.5% vs 60.0%，差 39.5 个百分点）。这直接说明单个 seed 不能作为最终结论。
- 上述低值已核对模型路径、53 维观测、实际摩擦=1.0、200 个环境 seed；属于真实策略表现，不是文件缺失或配置串用。

## 产物

- 结果目录：`outputs/evaluation_suites/formal_multiseed_n200_envseeds_10000_10199/`
- 逐回合 CSV / summary：`outputs/runs/formal_norm_<config>_seed<43|44>/evaluations/formal_multiseed_n200_envseeds_10000_10199_trainseed<seed>_<condition>_*`
- 逐 seed 结果：`per_training_seed.csv`（50 行）
- 跨 seed 聚合：`aggregate_across_training_seeds.csv`（25 行）
- 总结 JSON：`summary.json`（`training_seeds=[43,44]`）
- 摩擦曲线：`friction_curve_success_rate.png`
- 运行日志：`run.log`；逐任务日志：`logs/`（50 个，各 200 条 episode 行）
- 两 seed 快照：`two_seed_snapshot/`（含 SNAPSHOT_INFO.json 和 SHA256）
- 产物清单：`artifact_manifest.json`（50 项 summary/CSV 的 SHA256）
- 启动前另做了 1 回合预检（`formal_norm_pos6_k1_seed43/evaluations/preflight_seed43_pos6_k1_20261115_*`），只验证环境/加载/输出，未计入 50 项或 10000 回合。

## seed42 到达后的补齐方式

当前评估脚本没有 manifest 文件（dry-run 只留下 suite 目录、`suite.lock` 和空的 `logs/`，没有任务清单），也不禁止后续增加训练 seed；但它要求**同一个 suite 名称、同一协议**，且 `completed_evaluation_is_compatible` 会校验 episodes/base_seed/摩擦范围/video/threshold/normalization。注意：`validate_model_run` 只校验 completed、seed、v1、观测/history、fixed_v1、final steps≥500k，**不校验 Git SHA、ent_coef、device 或网络层**，所以 seed42 到达后必须像本次一样人工补核这几项。正确步骤：
1. 确认 seed42 五个模型已放入 `outputs/runs/formal_norm_*_seed42/`，并用同样字段核验。
2. 先把当前两 seed 的 `per_training_seed.csv`、`aggregate_across_training_seeds.csv`、`summary.json`、`friction_curve_success_rate.png` 保留好（本目录已有 `two_seed_snapshot/`）。
3. 用同一 suite 名重跑：`bash scripts/local.sh -m scripts.evaluate_formal_multiseed_suite --training-seed 42 --training-seed 43 --training-seed 44 --suite-name formal_multiseed_n200_envseeds_10000_10199`。脚本会跳过已完成的 43/44，只评估 42，然后写出三 seed 聚合。
4. 若改变 episodes、环境 seed 范围、摩擦条件或视频设置，脚本会对已有 43/44 结果报“settings different”，此时应使用新的 suite 名，不能覆盖本目录。
5. 三 seed 聚合完成后再写最终报告；本文件中的两 seed 表格不得直接当作最终表。

## 声明

- 本次未训练 seed42、未重训 seed43/44、未修改课程提交报告、未 push。
- 模型文件存在与“性能已验证”是两件事；以上是正式评估结果，但仍只是两 seed 证据。
