# Laya CPU vs Ascend NPU 推理速度对比报告

- 生成时间：2026-09-23T05:39:19
- 主机/平台：`f6163183d07e` / `Linux-5.10.0-136.12.0.86.r1526_92.hce2.aarch64-aarch64-with-glibc2.38`
- Python：`3.11.15`；PyTorch：`2.8.0+cpu`；torch_npu：`2.8.0.post4`；Transformers：`5.17.0`；CANN：`9.0.0`
- CPU：容器可见 `192` 个逻辑核；正式测试使用 `1` 线程（该容器中 1 线程最快，见附录）
- NPU：`Ascend910B1` × 2，测试使用 `npu:0`
- 模型：本地 `models/laya`、`models/laya-multilingual`、`models/laya-typed-decisions`
- 方法：每个模型/设备先 warmup 10 次，再正式计时 30 次；NPU 前后调用 `torch.npu.synchronize()`，使用 `perf_counter` 记录端到端 `Agent.predict` 延迟

## 1. 延迟与吞吐结果

| 模型 | 输入案例 | 输入 tokens | CPU 中位延迟 (ms) | NPU 中位延迟 (ms) | NPU 加速比 | CPU p95 (ms) | NPU p95 (ms) | CPU tokens/s | NPU tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| laya (English checkpoint) | English state / 4 questions | 348 | 3,135.817 | 46.388 | 67.60x | 4,179.948 | 48.650 | 110.98 | 7,501.99 |
| laya-multilingual | Hindi state / 4 questions | 236 | 1,259.462 | 37.285 | 33.78x | 1,633.621 | 37.597 | 187.38 | 6,329.59 |
| laya-typed-decisions | English state / 4 questions | 348 | 3,168.896 | 44.681 | 70.92x | 4,167.961 | 44.884 | 109.82 | 7,788.54 |

### 完整统计

| 设备/模型 | mean (ms) | median (ms) | std (ms) | CV | min (ms) | p90 (ms) | p95 (ms) | max (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU / laya (English checkpoint) | 3,320.289 | 3,135.817 | 369.641 | 11.13% | 3,030.687 | 3,794.078 | 4,179.948 | 4,297.259 |
| CPU / laya-multilingual | 1,305.336 | 1,259.462 | 150.647 | 11.54% | 1,196.886 | 1,599.054 | 1,633.621 | 1,695.790 |
| CPU / laya-typed-decisions | 3,386.192 | 3,168.896 | 437.744 | 12.93% | 2,929.790 | 4,160.361 | 4,167.961 | 4,223.324 |
| NPU / laya (English checkpoint) | 46.906 | 46.388 | 1.074 | 2.29% | 45.354 | 48.382 | 48.650 | 48.881 |
| NPU / laya-multilingual | 37.311 | 37.285 | 0.172 | 0.46% | 36.722 | 37.430 | 37.597 | 37.743 |
| NPU / laya-typed-decisions | 44.671 | 44.681 | 0.129 | 0.29% | 44.359 | 44.789 | 44.884 | 44.987 |

## 2. 主要结论

- **laya (English checkpoint)**：CPU 3,135.817 ms -> NPU 46.388 ms，加速 **67.60x**，中位延迟降低 98.5%。
- **laya-multilingual**：CPU 1,259.462 ms -> NPU 37.285 ms，加速 **33.78x**，中位延迟降低 97.0%。
- **laya-typed-decisions**：CPU 3,168.896 ms -> NPU 44.681 ms，加速 **70.92x**，中位延迟降低 98.6%。

- 三个 checkpoint 在 NPU 上均进入 **37–47 ms** 级别，满足实时决策场景；CPU 单线程为 **1.26–3.17 s**。
- NPU 延迟抖动明显更小：NPU CV 约 **0.29%–2.29%**，CPU CV 约 **11.1%–12.9%**，说明 NPU 在 warmup 后非常稳定。
- `laya-multilingual` 因编码器更小（mmBERT-base，322M）所以 CPU/NPU 都更快；`laya` 与 `typed-decisions` 为 ModernBERT-large（421M），绝对延迟接近。

## 3. 公平性说明

- **相同模型权重、相同输入、相同问题、相同 `predict` 端到端流程**；只有 `device` 不同。
- **模型加载时间不计入推理延迟**，warmup 后才开始计时；本机每个模型冷加载约 51–55 s。
- **先 warmup 10 次**：消除 NPU 首次算子编译、显存分配、CPU 频率爬升、页表/缓存带来的首轮偏差。
- NPU 每次计时前后均 `torch.npu.synchronize()`，确保异步 kernel 真正执行完；CPU 本身同步。
- CPU 与 NPU 都使用 **float32** 权重，没有启用 NPU bf16 autocast，避免精度策略不同导致比较不公。
- CPU 线程数选择该环境实测最优的 **1 线程**；多线程结果见附录，不代表正式结果。
- NPU 使用与 `infer-npu.py` 相同的等价 SDPA 决策头实现，避免当前 torch_npu 缺少 `aten::_transformer_encoder_layer_fwd` 导致整层回退 CPU；已验证输出概率与原生实现一致。

## 4. CPU 线程数扫描（附录）

使用 `laya (English)`、同一输入，仅用于选择 CPU 配置；warmup=2、runs=5，因此数值仅用于横向排序。

| CPU 线程数 | 中位延迟 (ms) | 相对 1 线程 |
|---:|---:|---:|
| 1 | 2,767.4 | 1.00x |
| 2 | 5,826.3 | 2.11x |
| 4 | 5,318.5 | 1.92x |
| 8 | 5,039.9 | 1.82x |
| 16 | 4,754.0 | 1.72x |
| 32 | 4,709.2 | 1.70x |
| 64 | 3,951.3 | 1.43x |
| 128 | 3,656.6 | 1.32x |

该容器虽然暴露 192 个逻辑核，但 PyTorch CPU 推理在 1 线程时最快；增加线程反而变慢，说明 CPU 侧结果受当前虚拟化/线程调度环境影响，不应直接外推到物理 CPU 服务器。

## 5. 复现命令

```bash
.venv/bin/python bench_cpu_vs_npu.py \
  --device both \
  --warmup 10 \
  --runs 30 \
  --cpu-threads 1 \
  --output benchmarks/cpu_vs_npu_benchmark.json
```

完整原始数据：`benchmarks/cpu_vs_npu_benchmark.json`；线程扫描数据：`benchmarks/cpu_thread_sweep.json`。

## 6. 限制

- 本次为 **batch=1、单条 state、4 个问题** 的端到端延迟测试，不代表高并发吞吐。
- 结果依赖 `torch_npu` / CANN / 驱动版本和 NPU 固件；不同版本可能不同。
- CPU 结果受容器 CPU 配额、NUMA、线程调度影响很大；如果是裸金属 CPU，需要重新测。
- NPU 使用 `npu:0`，测试时未与其他 NPU 任务共享该卡；生产环境建议按实际卡、并发和 batch 重新压测。
