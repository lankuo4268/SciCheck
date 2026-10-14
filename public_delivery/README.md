# SciCheck

面向结构化科学计算任务的独立验证与受控修复演示。

计算工具正常返回，仍需检查候选是否满足原始精度要求。SciCheck 保留候选、独立判定、受控修复、预算及回执，覆盖积分、求根、有理数线性系统和受支持的线性标量 ODE。

## 查看演示

下载仓库后，用浏览器打开 [scicheck_demo.html](scicheck_demo.html)。点击右上角“演示模式”适合录屏；ODE可切换三个问题。页面离线回放已有记录，不调用模型、不启动科学计算。GitHub文件预览显示HTML源码，请下载后打开。

## 运行源码

已验证计算环境为 Python 3.13.9、NumPy 2.3.5、SciPy 1.16.3。源码位于 source/，样例位于 examples/。

```sh
python -m pip install -r requirements.txt
python source/scicheck_cli.py examples/piecewise_narrow_contrib.json --compare
python -B -m unittest discover -s source -v
```



## 已有记录与口径

当前分层索引见 [results/acceptance_candidate_run_index_v2_20261015063151.json](results/acceptance_candidate_run_index_v2_20261015063151.json)。11份场景原始记录引用26个不同运行ID：

| 证据层 | 运行ID数 | 原记录判定 |
|---|---:|---|
| 有执行结果记录 | 23 | 通过15、未通过8 |
| 仅有回执验证记录 | 2 | 通过1、未通过1 |
| 仅有幂等重放引用 | 1 | 原运行判定缺失 |

这些是开发记录的证据分类，不是模型效果通过率。回执验证和重放不增加候选执行次数。拒绝、执行异常及终态观察在索引中另列。独立路线的任务归属如仅有源码关联证据，索引明确标注。

另有AGH宿主实际五步闭环：读取结构、普通积分判错、验证错误回执、断点积分修复、验证通过回执。该宿主记录与本地场景记录分开说明，不重复混入上述分母。

公开包不附原始私有输入JSON、数据库、密钥、账户和会话库。统计生成与校验脚本原样保留为来源材料；重新聚合历史统计需要工作区中的11份原始输入和相应文件结构，不能在缺少输入时声称复现成功。

## 界面检查

当前展示页通过独立Chrome的1440px与390px模拟宽度检查，共114项：案例切换、ODE子案例、回放、切换停止定时器、JSON下载一致性、复制失败反馈、来源哈希、演示模式及运行时错误检查。见 [docs/interface_checks.json](docs/interface_checks.json)。这不是实体手机或全浏览器兼容性认证。

## 实现来源与边界

- AGH对话中的模型：Agnes / agnes-3.0-flash。业务源码由AGH逐步生成与修正；当前目录整理阶段只逐字节复制已有业务源码，见 [docs/source_provenance.json](docs/source_provenance.json)。
- 展示页面的视觉与交互由AI维护侧优化，内嵌原始DATA保持不变；不把界面优化描述成科学计算或模型策略改进。
- AGH官方宿主参考：https://github.com/AgnesAI-Labs/agnes-harness/tree/a39272bb6bcb3109584e1bd3abac595789ae365d
- 候选路线使用NumPy/SciPy；独立参考使用结构化解析计算、Fraction或受支持ODE的Decimal包络。第三方组件遵循各自许可证，本次未另行授予项目级许可证。
- 插件源码位于source/agh-plugin/；仓库不附AGH运行时、真实绑定或账户。安装、信任、启用和受信任务登记需在本机AGH另行配置。
- 科学模式的宿主模型请求预算强制钩子、非业务工具限制与操作系统隔离尚未完整验证；实际费用保持UNKNOWN。



