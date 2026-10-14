# 本轮统计验收结果

## 生成产物
- 新版统计文件：`private_runs/acceptance_candidate_run_index_v2_20261015063151.json`
- 生成脚本：`_build_candidate_run_statistics.py`
- 独立校验脚本：`_check_candidate_run_statistics.py`
- 旧索引已保留：`private_runs/acceptance_run_index_20261015060904.json`
- 旧统计文件已保留：`private_runs/acceptance_statistics_relayered_20261015055010.json`

## 分层结果
- 独立 run_id 总数：26
- 分层：
  - `execution_result_record`: 23
  - `replay_reference_only`: 1
  - `receipt_verification_only`: 2
- 判定：
  - `execution_result_record`: VERIFIED_PASS 15, VERIFIED_INCORRECT 8
  - `replay_reference_only`: VERDICT_MISSING 1
  - `receipt_verification_only`: VERIFIED_PASS 1, VERIFIED_INCORRECT 1
- 说明：
  - 候选执行记录数不等于 receipt/replay 引用数，也不等于模型效果通过率。
  - 这轮没有新增 solve、model call 或 AGH host tool call，fee 保持 `UNKNOWN`。

## 四项修复
1. 区分“运行 ID 索引”和“候选执行记录”：
   - 已按 `run_id` 合并所有引用，并按证据类型分层。
   - replay 和 verify/receipt 不再扩大候选执行记录总数。
2. 修正 task 归属：
   - 未把所有事件无条件套顶层 `task`。
   - 三条独立 route 单独记录：
     - `run-9653cf4d...` -> `task-root-direct-x-1-cubed`
     - `run-4028708f...` -> `task-ls-required-decimal-high`
     - `run-e1390ec3...` -> `task-ls-ultra-tight-decimal-high`
   - 对应代码证据来自 `scicheck_task_cli.py` 的独立注册调用。
3. 修正嵌套字段与 0 值：
   - `run-c8181bcd...` 现在从 `sections.active_verify_after_restart.verify` 读取。
   - 保留 `candidate=0.0`、`task_id=acc-restart-active`、`verdict=VERIFIED_INCORRECT`。
   - 校验通过 `candidate_zero_preserved_for_run_c8181bcd`。
4. 重构 stop/reject/exception index：
   - 拒绝事件：10 条
   - 无候选执行异常：1 条
   - 终态观察：6 条
   - 预算耗尽、终端观察、拒绝调用已分开计数，不再把终态观察直接当作拒绝事件。

## 检查
`_check_candidate_run_statistics.py` 全部通过：
- 11 份输入文件哈希一致
- 旧索引保留且哈希未变
- 全部引用 JSON 路径可解析
- 26 个独立 run_id 完整覆盖
- 分层互斥且数量相加正确
- 判定分类可按层对账
- 三条独立 route task attribution 已记录
- `candidate=0.0` 已正确保留
- 拒绝事件逐条列出
- 重复生成可复现

## 仍未完成
- 页面修复
- 浏览器验证
- 交付目录打包
- 科学计算重跑
