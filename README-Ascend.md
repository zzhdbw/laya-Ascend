# Laya Ascend 推理指南

在 Ascend 910B1 + CANN 9.0.0 环境中安装依赖并运行 Laya 文本决策推理。

## 环境版本

| 组件 | 版本 |
|---|---|
| NPU | Ascend 910B1 × 2 |
| NPU 驱动 | 25.5.1 |
| CANN Toolkit / OPP | 9.0.0 |
| Python | 3.11.15 |
| torch | 2.8.0+cpu |
| torch-npu | 2.8.0.post4 |
| transformers | 5.17.0 |
| safetensors | 0.8.0 |
| numpy | 2.4.6 |
| huggingface-hub | 1.32.0 |
| tokenizers | 0.23.2 |
| modelscope | 1.40.1 |
| laya | 0.3.5 |

CANN 路径：

```text
/usr/local/Ascend/cann-9.0.0
```

torch 与 torch-npu 必须匹配：

```text
CANN 9.0.0 -> torch 2.8.0 -> torch-npu 2.8.0.post4
```

## 前置条件

1. 已安装 Ascend NPU 驱动、固件和 CANN。
2. 已执行 CANN 环境脚本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

3. 确认设备可用：

```bash
npu-smi info
```

应能看到 `910B1`，状态为 `OK`。

## 初始化

### 1. 进入项目

```bash
cd /data/zzh/laya-Ascend
```

### 2. 创建虚拟环境

当前环境使用 uv：

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 3. 安装依赖

`pyproject.toml` 不会自动安装 torch，需要按 Ascend 环境手动安装。

#### uv

```bash
# 文本依赖和项目本身
uv pip install torch-npu==2.8.0.post4
uv pip install -e .
uv pip install modelscope
```

### 4. 验证环境

```bash
.venv/bin/python - <<'PY'
import torch
import torch_npu

print("torch     :", torch.__version__)
print("torch_npu :", torch_npu.__version__)
print("available :", torch.npu.is_available())
print("count     :", torch.npu.device_count())

for i in range(torch.npu.device_count()):
    print(i, torch.npu.get_device_name(i))
PY
```

预期结果：

```text
torch     : 2.8.0+cpu
torch_npu : 2.8.0.post4
available : True
count     : 2
0 Ascend910B1
1 Ascend910B1
```

### 5. 下载模型

| 模型 | 目录 |
|---|---|
| `convaiinnovations/laya` | `models/laya` |
| `convaiinnovations/laya-multilingual` | `models/laya-multilingual` |
| `convaiinnovations/laya-typed-decisions` | `models/laya-typed-decisions` |

ModelScope：

```bash
cd models
bash download.sh
```

Hugging Face：

```bash
# 国内网络可设置
export HF_ENDPOINT=https://hf-mirror.com

cd models
hf download convaiinnovations/laya --local-dir ./laya
hf download convaiinnovations/laya-multilingual --local-dir ./laya-multilingual
hf download convaiinnovations/laya-typed-decisions --local-dir ./laya-typed-decisions
```


## 推理

### CPU / 默认设备

```bash
.venv/bin/python infer.py
```

`infer.py` 使用本地三个 checkpoint，并自动进行英文/多语言路由。

### Ascend NPU

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
.venv/bin/python infer-npu.py
```

指定 NPU：

```bash
LAYA_NPU_DEVICE=npu:0 .venv/bin/python infer-npu.py
```

按需加载模型，减少启动占用：

```bash
LAYA_NPU_PRELOAD=0 .venv/bin/python infer-npu.py
```

输出示例：

```text
NPU device   : npu:0
Devices      : 1 x Ascend910B1
Preload      : True

[English]
  Department : billing
  Routing    : english
  Device     : npu:0

[Hindi]
  Department : billing
  Routing    : multilingual
  Device     : npu:0

[Explicit typed-decisions]
  Department : billing
  Routing    : typed-decisions
  Device     : npu:0
```

### NPU 决策头适配

当前 `torch_npu` 不支持 `aten::_transformer_encoder_layer_fwd`，Laya 决策头可能整层回退 CPU。

`infer-npu.py` 使用等价的 `F.scaled_dot_product_attention` 实现该层 forward，权重不变，输出结果一致。

## 性能

warmup 10 次、正式测量 30 次：

| 模型 | CPU 中位延迟 | NPU 中位延迟 | 加速比 |
|---|---:|---:|---:|
| laya / English | 3135.817 ms | 46.388 ms | 67.60x |
| laya-multilingual | 1259.462 ms | 37.285 ms | 33.78x |
| laya-typed-decisions | 3168.896 ms | 44.681 ms | 70.92x |

复现：

```bash
.venv/bin/python bench_cpu_vs_npu.py \
  --device both \
  --warmup 10 \
  --runs 30 \
  --cpu-threads 1 \
  --output benchmarks/cpu_vs_npu_benchmark.json
```

详细结果见 `benchmarks/cpu_vs_npu_report.md`。

## 快速开始

```bash
cd /data/zzh/laya-Ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
.venv/bin/python infer-npu.py
```
