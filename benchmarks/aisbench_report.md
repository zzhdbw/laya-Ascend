# AISBench / Ascend 310P3 实测

日期：2026-09-24。工作目录：`/root/code/wwh/laya-Ascend`，conda 环境：`ascend`。

## 环境与隔离

- Ascend 310P3，只使用 **Device 7（NPU 5 / Chip 1）**；启动前确认无运行进程。
- Driver 26.1.1，CANN 9.1.0（任务环境固定绝对路径，不修改系统 CANN 链接）。
- Python 3.11.16，torch 2.8.0+cpu，transformers 5.17.0。
- NumPy 1.26.4，ONNX 1.17.0，ONNXRuntime 1.23.2。
- `aclruntime` / `ais_bench` 0.0.2，使用 Ascend/tools **Gitee** 源码 `8da7f6455bb122ca38b50e625fb7751c7e190ed1`，未修改该源码；同版本号的早期 wheel API 不同。
- 未安装 torch-npu；模型前向完整运行于 OM，不存在 torch CPU 回退。
- 任务代码、模型、缓存、日志保存在工作目录；依赖安装在指定 ascend 环境。未修改其他服务、驱动或系统配置。

## 模型产物

三个模型均已执行 ONNX 导出、ONNXRuntime logits 校验和 ATC 编译：

| 模型目录 | 静态 batch | 序列容量 | 选项容量 |
|---|---:|---:|---:|
| `models/aisbench/english` | 1 | 512 | 32 |
| `models/aisbench/multilingual` | 1 | 1024 | 32 |
| `models/aisbench/typed-decisions` | 1 | 1024 | 32 |

SoC：`Ascend310P3`；编译精度：`allow_fp32_to_fp16`。每个目录包含 `model.om`、tokenizer、运行时配置及实际 ATC 命令；ONNX 也保留在该目录。权重/ONNX/OM 不纳入 git。

## 精度 smoke regression

同一 checkpoint 的 CPU FP32 输出对比 OM：混合 choice/score/noul、单选项、五问题分块，共三组请求/模型。对比所有答案数值、选择及 token usage；阈值 0.02。

| 模型 | 最大绝对误差 | 选择一致 | token usage 一致 |
|---|---:|---|---|
| English | 0.0022 | 是 | 是 |
| multilingual | 0.0002 | 是 | 是 |
| typed-decisions | 0.0005 | 是 | 是 |

完整记录：[aisbench_accuracy.json](aisbench_accuracy.json)。这是集成回归，不代表完整数据集的模型质量评测。

## 端到端耗时

每次请求四个问题；warmup 10 次，正式测量 30 次。包括分词、填充、四次 batch=1 的 OM 执行、传输和后处理，不包括首次模型加载。

| 模型 | 中位延迟 (ms/请求) | P95 (ms/请求) |
|---|---:|---:|
| English | 220.303 | 222.403 |
| multilingual | 336.701 | 338.812 |
| typed-decisions | 725.562 | 726.995 |

原始样本、实际运行设备、静态形状和 token 数见 [aisbench_benchmark.json](aisbench_benchmark.json)。

**本次未测同机 torch-npu（310P 走 OM 路径），不据此宣称比 README 中另一台 910B1 更快。** 静态完整长度有明显填充开销；短请求可另导出较小 `--seq-len`，多问题请求可尝试较大 `--batch-size`，并重新验证精度/性能。当前结果不是极限调优值。

## 应用与回归

已通过：

- English / Hindi 自动路由，typed-decisions 显式路由。
- Snake 终端 3 步，Tetris 终端 2 个方块。
- 两个 Web 示例的 `/api/info`、新游戏、执行一步接口；真实 OM，HTTP 200。测试只监听临时 loopback 端口，结束后关闭。
- 原有 router、criteria、email、download、shortlist、decision_model、packaging 测试。
- 新增 10 个 AISBench 适配测试、3 个真实 ONNX 导出测试，以及源码字节编译检查。
- AISBench 官方 CLI `--help` 可用；各实际推理进程正常退出，无退出时崩溃。

最终检查退出码为 0。`npu-smi info` 确认测试完成后无残留 NPU 进程；未留下 Web 服务。

## 在本次机器复现

```bash
cd /root/code/wwh/laya-Ascend
source .work/env.sh  # 激活 ascend，固定 CANN 9.1 与任务缓存目录
python infer-aisbench.py --device-id 7
python validate-aisbench.py --device-id 7 --output benchmarks/aisbench_accuracy.json
python bench_cpu_vs_npu.py --device aisbench --aisbench-device 7 \
  --cpu-threads 4 --warmup 10 --runs 30 --output benchmarks/aisbench_benchmark.json
```

再次运行前请确认 Device 7 仍空闲。完整使用说明见 [README-AISBench.md](../README-AISBench.md)。任务级日志在 `.work/logs/`，环境包清单在 `.work/packages-after.txt`。
