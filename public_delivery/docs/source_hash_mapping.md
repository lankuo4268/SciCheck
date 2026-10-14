# 统计来源与哈希映射

## 新版统计文件
- 工作区：`private_runs/acceptance_candidate_run_index_v2_20261015063151.json`
- 交付副本：`results/acceptance_candidate_run_index_v2_20261015063151.json`
- 生成脚本：`source/build_candidate_run_statistics.py`
- 校验脚本：`source/check_candidate_run_statistics.py`

## 输入文件（保留在工作区 `private_runs/`，未放入本包）
- `acc_integration_normal_20261015042620.json`
- `acc_integration_zero_candidate_20261015042620.json`
- `acc_root_normal_20261015042620.json`
- `acc_root_tight_20261015042620.json`
- `acc_linsys_normal_20261015042620.json`
- `acc_linsys_no_unique_20261015042620.json`
- `acc_linsys_ultra_tight_20261015042620.json`
- `ode_stability_feec260053604a99bd12b37b25a40c2f.json`
- `acc_cross_session_20261015042632.json`
- `acc_budget_exhaust_20261015042633.json`
- `acc_restart_20261015042635.json`

## 旧版对照
- 旧运行索引：`private_runs/acceptance_run_index_20261015060904.json`
- 旧分层统计：`private_runs/acceptance_statistics_relayered_20261015055010.json`

## 哈希方法
- 输入文件：逐文件 SHA-256。
- 统计文件：生成脚本记录确定性内容哈希（排除 `generated_at` 时间戳）。
- 校验脚本会重新计算输入 SHA-256 并与生成结果比对。

## 边界
- 未修改原始记录。
- 未修改数据库或密钥。
- 未将分层统计描述为模型效果通过率。


