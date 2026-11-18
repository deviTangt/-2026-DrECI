# 第14组 Dr.ECI 事件因果识别复现

机器智能课程设计

## 完成内容与实测结果

- 完整四角色流程：Causal Explorer、Mediator Detector、Direct / Indirect Reasoner。
- 使用官方 ESC CSV 全部9,978行，按话题分为训练6,052、验证1,905、测试2,021对。
- 同模型零样本对照：E0直接提示、E1完整Dr.ECI(q=1)、E2只改q=2。
- 独立测试总体F1：E0 72.69%；E1 72.04%；E2 70.04%。
- 按返回token和公开价估算的全部调用费用上界：¥7.1329，授权上限¥20。
- 两个真实案例、五个错例、可重放与实时演示。

本方法是大模型调用流程，没有梯度训练。训练划分用于接口调试，按照课程要求提交提示、原始响应和解析轨迹。实验完成于2026年11月3日；课程报告、LaTeX源码与PPT另行提交。

## 代码与实验记录

本仓库保存本组复现代码、冻结配置、官方提示、数据来源审计、逐对预测、推理轨迹及原始 API 响应。课程提交包另行提供 LaTeX 源码、报告 PDF 和汇报 PPT。

首先查看 `results/comparison.csv`、`results/summary.json` 和 `data_info/audit.json`；运行 `code/verify_delivery.py` 可离线核对指标与记录完整性。

## 环境安装（fish命令）

下载仓库后，在仓库根目录执行：

```fish
python -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
```

实测环境：CachyOS Linux、Python 3.14，推理由DeepSeek API提供，本机GPU不参与模型推理。

## 不花费API费用的验收与演示

```fish
.venv/bin/python code/verify_delivery.py
.venv/bin/python code/demo.py --list
.venv/bin/python code/demo.py --sample-id ESC-CSV-04255
.venv/bin/python code/demo.py --interactive
```

`demo.py`默认明确显示“真实实验记录重放”，逐阶段展示文本、问法回答、中介、推理和最终标签。记录有原始API请求/响应ID，可检查来源；它不是新调用。

## 实时API演示

将`.env.example`复制为`.env`并在本地填写自己的`DEEPSEEK_API_KEY`，或在fish中配置环境变量。不要将密钥写入报告或上传。

```fish
cp .env.example .env
# 在编辑器中填写 .env 后：
.venv/bin/python code/demo.py --live --sample-id ESC-CSV-00046
```

实时演示使用单独保留的训练样本，不改变测试结果。如果这个样本已调用，完整相同请求会复用缓存，程序会显示“实时API/缓存调用”。需要换一个训练样本时，可查看`data_info/demo_train.json`中的ID；不会暗中换成测试样本。

## 数据准备和实验复算

仓库不包含完整数据集、虚拟环境和API密钥。公开数据与原始代码下载地址、版本哈希记录在`data_info/audit.json`。

```fish
.venv/bin/python code/fetch_sources.py
.venv/bin/python code/prepare_data.py
.venv/bin/python code/test_integrity.py
.venv/bin/python code/experiment.py --split validation --workers 16
.venv/bin/python code/experiment.py --split test --workers 16
.venv/bin/python code/analyze_results.py
```

本仓库包含已冻结配置与全部实验响应，以上命令会复用相同请求；正式新实验应复制到新目录，明确更改实验ID、模型和预算，并重新走验证选参再冻结的流程。`freeze_validation.py`拒绝覆盖已冻结文件。不要根据本次测试结果修改本次配置后仍称其为独立测试。

## 协议与实现说明

- 任务：Cause-Effect/NONE二分类，识别存在性；方向指标不适用，不输出虚构概率。
- 论文主要模型为PaLM2，当前复现为DeepSeek-V4.1-Flash，属于替换底层模型的方法复现。
- 数据候选来自作者发布的完整CSV，但本次使用自定义话题隔离划分，不是原论文测试协议。
- 官方仓库默认只跑前2项，且存在回答字段名和固定字符串解析问题；本实现修复执行问题，保留核心提示与八规则语义。
- 官方代码用一个条件式联合提示实现直接/间接推理。本实现保留联合提示，并根据中介是否为空显式记录两条路径。
- E0与探索第一问请求完全一致；E2只变q，因此可复用相同阶段的真实回答。逐方法费用是所需请求的等效成本，不能相加当作实际扣费。
- 每个原始样本都纳入评价，包括重复输入。3组输入标签冲突原样保留；集合之间文档与输入哈希重叠均为0。
- 失败不丢弃。存在性正例弃答计入FN，负例弃答计入FP，覆盖率单列；原始JSON格式恢复另有记录。
- 时间口径：逐方法平均串行API时间来自各阶段实测耗时之和；联合并发墙钟时间记录在`results/run_summary.json`和运行日志。

## 目录

```text
code/                         预处理、实验、评价、演示、验证脚本
configs/                      E0/E1/E2、划分、运行和冻结配置
prompts/                      官方三阶段提示文件
data_info/                    来源、统计、哈希、图检查、少量演示样本
results/E0 E1 E2/              验证与测试逐行预测、指标
results/api_calls/            实际请求、原始响应、用量和耗时
results/traces/               四角色流程轨迹
results/figures/              测试图表和TD方法图
results/cases_selected.json   两个案例与五个错例
```

## 来源

论文：https://aclanthology.org/2025.coling-main.628/

官方实现：https://github.com/DMIRLAB-Group/Dr.ECI

原始ESC：https://github.com/tommasoc80/EventStoryLine

所有原论文数字与本次实测数字分开标记；引用和配置差异详见报告。
