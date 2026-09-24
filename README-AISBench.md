# Laya AISBench 离线推理

把 **English / multilingual / typed-decisions 三个完整决策模型**（编码器 + 决策头）导出为 ONNX，经 CANN ATC 编译为 OM，再由 `ais_bench.infer.interface.InferSession` 执行。CPU/torch-npu 原有路径保留。

本次 310P3 的实测结果与限制见 [AISBench 测试报告](benchmarks/aisbench_report.md)。

这里的 AISBench 指 Ascend **OM 推理工具**，不是大模型评测框架。推理不需要 `torch_npu`，但分词、输入整理和后处理仍使用 CPU PyTorch/NumPy。

## 环境

```bash
conda activate ascend
source /usr/local/Ascend/cann/set_env.sh  # 按实际 CANN 路径调整
python -m pip install -e '.[aisbench-export]'
```

另外安装与 Python、CPU 架构和 CANN 匹配的 `aclruntime`、`ais_bench` 推理包，参见 [Ascend 官方说明](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)。不要把其他同名评测工具当成推理包。PyTorch 请按机器安装 CPU 版本；导出使用 torch 2.8、transformers 5.17 验证。

若只有旧 wheel，可在任务目录构建当前官方源码（需要 C++ 编译器和 zlib 开发头文件）：

```bash
mkdir -p .work
git clone https://gitee.com/ascend/tools.git .work/ascend-tools
# 本次验证的源码版本；aclruntime/ais_bench 的包版本号仍为 0.0.2
git -C .work/ascend-tools checkout 8da7f6455bb122ca38b50e625fb7751c7e190ed1
export CANN_PATH=/usr/local/Ascend/cann-9.1.0  # 使用实际且完整的 CANN 路径
python -m pip install 'pybind11==2.13.6' attrs tqdm
# conda 环境缺少 zlib.h 时，仅在当前环境安装：conda install zlib
python -m pip install --no-build-isolation \
  .work/ascend-tools/ais-bench_workload/tool/ais_bench/backend \
  .work/ascend-tools/ais-bench_workload/tool/ais_bench
```

CANN 的 ATC Python 依赖也必须在当前环境可用。CANN 环境若需要旧版 NumPy/protobuf，可使用 `numpy==1.26.4 onnx==1.17.0 onnxruntime==1.23.2 protobuf==3.20.3`，不要覆盖系统 Python 或修改其他环境。

## 1. 准备三个 checkpoint

按 [Ascend 指南](README-Ascend.md) 下载到：

```text
models/laya/
models/laya-multilingual/
models/laya-typed-decisions/
```

## 2. 导出与编译

导出仅使用 CPU，不占用 NPU：

```bash
python export-aisbench.py
```

默认导出三个模型；单独导出可传 `--models english` 等。产物为：

```text
models/aisbench/{english,multilingual,typed-decisions}/
  model.onnx
  tokenizer/
  aisbench_config.json
```

导出器使用显式全局/局部注意力掩码及可导出的决策头，不全局 monkeypatch PyTorch。它会检查 ONNX，并用 ONNXRuntime 对比两路 logits；检查失败不会发布运行时配置。输出目录必须为空，避免新 tokenizer/config 搭配旧 OM。

在目标 CANN 环境编译，**SoC 必须匹配硬件**：

```bash
# 310P3 示例；其他设备换成实际 SoC，例如 Ascend910B1
python export-aisbench.py --compile-only --soc-version Ascend310P3 \
  --precision-mode allow_fp32_to_fp16
```

也可一次执行导出与编译：`--compile --soc-version ...`。默认编译精度为 `must_keep_origin_dtype`；310P 的浮点算子可使用上述 `allow_fp32_to_fp16`，然后必须执行下方精度验证。生成 `model.om` 与 `atc_command.json`。

### 静态形状与限制

- 默认 `batch_size=1`、`max_options=32`，序列长度为 checkpoint 的 `max_len`。
- 任意问题数量自动分块；最后不足一个 batch 时复制有效行，防止全掩码注意力产生 NaN。只返回真实问题，token 计数不含填充。
- `--batch-size 4` 可服务多问题请求，但单问题也会按该 batch 执行。
- `--seq-len 256` 可减少短请求计算量，**超过导出容量的请求会明确报错，不会额外静默截断**；不要用短 OM 替代需要长上下文的模型。
- 超过 32 个选项时重新导出 `--max-options N`，或使用外部 embedding 做 shortlist。
- `choice`（含单选项）、`score`、`noul`、温度校准、概率、置信度、action probability 和 token usage 复用原 Agent 逻辑。

## 3. 单卡推理与路由

先运行 `npu-smi info` 确认所选 **Device ID** 空闲。一个进程的全部模型使用同一个显式 ACL device id，不自动选择或占用其他卡。注意表中的 NPU/Chip/Device 编号不同；不要假设 `ASCEND_RT_VISIBLE_DEVICES` 在所有 ACL 版本都重映射编号。

```bash
export LAYA_AISBENCH_DEVICE=7  # 示例；请换成你获准使用的空闲 Device ID
python infer-aisbench.py --device-id "$LAYA_AISBENCH_DEVICE"
```

Python API：

```python
from laya import load, Router

with load("models/aisbench/english", backend="aisbench", device="npu:7") as agent:
    result = agent.predict("Please refund the duplicate charge.", questions)

router = Router(
    models={name: "models/aisbench/" + name
            for name in ("english", "multilingual", "typed-decisions")},
    backend="aisbench", device="npu:7", max_loaded=1,
)
result = router.predict("请退还重复扣款。", questions)  # multilingual
result = router.predict(state, questions, model="typed-decisions")
router.unload()
```

AISBench 不会加载 `model.safetensors`，也不会在 NPU 出错时静默回退 CPU。需要当前 Gitee 版本的 `free_resource()` / 静态 `InferSession.finalize()` API（旧 GitHub 镜像/早期同版本号 wheel 不满足此要求）。每个 session 的执行/释放有锁；`close()` 只释放自己的模型，不调用全局 ACL finalize。进程退出时统一关闭本库的 session，再调用 `InferSession.finalize()`，避免 CANN 原生库晚析构崩溃。需要 encoder embedding 的 `embed_fn_from_agent` 仍是 torch 专用；`predict_shortlist` 可以使用外部 `embed_fn` 与 AISBench Agent 配合。

## 4. 精度与性能

```bash
python validate-aisbench.py --device-id "$LAYA_AISBENCH_DEVICE" \
  --output benchmarks/aisbench_accuracy.json

python bench_cpu_vs_npu.py --device aisbench --aisbench-device "$LAYA_AISBENCH_DEVICE" \
  --warmup 10 --runs 30 --output benchmarks/aisbench_benchmark.json
```

精度脚本使用相同的 CPU checkpoint 对比三类问题、单选项及多批次，检查选择、所有数值字段和 token usage；默认最大绝对误差为 0.02。该检查是 smoke regression，不代表完整数据集精度评测。

性能包含分词、静态填充、H2D、OM 执行、D2H、后处理。AISBench `infer()` 返回 host 数组后再停止计时。静态长序列存在填充开销，不能直接拿不同 NPU/长度/批量的结果宣称比 torch-npu 更快。原 CPU/NPU benchmark 命令不变；CPU/torch-npu 与 AISBench 分进程测量，避免两种运行时的初始化与清理互相干扰。310P 环境使用 CPU/AISBench 模式。

## 5. Snake / Tetris

两个示例的终端和 Web 入口均支持 `--backend aisbench`，策略和安全规则不变：

```bash
python examples/snake/play.py --backend aisbench --device npu:7 \
  --model models/aisbench/multilingual --steps 3 --delay 0
python examples/tetris/play.py --backend aisbench --device npu:7 \
  --model models/aisbench/multilingual --pieces 2 --delay 0

# 如需 Web 界面，先确认端口空闲；默认仅监听 127.0.0.1
python examples/snake/server.py --backend aisbench --device npu:7 \
  --model models/aisbench/multilingual --port 8010
python examples/tetris/server.py --backend aisbench --device npu:7 \
  --model models/aisbench/multilingual --port 8020
```

## 离线回归测试

```bash
python tests/test_aisbench.py
python tests/test_aisbench_export.py
```

第一组只替换 ACL transport，真实测试模型、分词、校准、分块、路由及资源释放；第二组实际导出两种 RoPE 配置的微型 ModernBERT 并运行 ONNXRuntime，不需要 NPU 或下载权重。
