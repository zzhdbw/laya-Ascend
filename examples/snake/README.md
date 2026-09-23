# Laya 贪吃蛇示例

这是一个使用 Laya 做实时决策的贪吃蛇示例，支持 CPU 和 Ascend NPU。每一步蛇的移动都由 Laya 对三个问题推理得到：

- `move`：四个方向中选哪个最安全且最接近食物
- `risk`：当前是否存在安全路线
- `food`：当前食物是否可达

外层还有一个确定性的循环安全护栏，只在模型提议不安全时纠正动作。

## 运行环境

先在仓库根目录安装好依赖：

```bash
cd /data/zzh/laya-Ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

默认使用 `models/laya-multilingual`，也可以传入任意本地 checkpoint：

```bash
--model models/laya
--model models/laya-typed-decisions
```

## 网页可视化

启动服务：

```bash
# CPU
.venv/bin/python examples/snake/server.py --device cpu --port 8010

# Ascend NPU
.venv/bin/python examples/snake/server.py --device npu:0 --port 8010
```

浏览器打开：

```text
http://127.0.0.1:8010
```

如果服务需要从其他机器访问：

```bash
.venv/bin/python examples/snake/server.py --device npu:0 --host 0.0.0.0 --port 8010
```

页面提供：

- Canvas 渲染贪吃蛇棋盘
- 新开一局 / 暂停 / 继续
- 速度调节
- 安全护栏开关
- compact / detailed 提示词切换
- 四个方向的模型概率
- 死路风险、食物可达性
- 得分、步数、护栏干预次数、单步推理耗时

## 终端运行

不想开浏览器时可以直接在终端跑：

```bash
.venv/bin/python examples/snake/play.py --device cpu --steps 20
.venv/bin/python examples/snake/play.py --device npu:0 --steps 50
```

常用参数：

```text
--model         checkpoint 路径，默认 models/laya-multilingual
--device        auto / cpu / npu:0 / npu:1
--width         棋盘宽度，默认 24
--height        棋盘高度，默认 16
--seed          随机种子
--steps         最多执行步数
--prompt        compact / detailed
--no-guard      关闭安全护栏
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `game.py` | 确定性贪吃蛇规则、食物生成和循环安全规划器 |
| `policy.py` | 调用 Laya 做移动、风险、食物可达三个问题的决策 |
| `play.py` | 终端 ASCII 渲染和自动运行 |
| `server.py` | 标准库 HTTP 服务，提供 `/api/snake/new` 和 `/api/snake/step` |
| `static/index.html` | 网页 Canvas 可视化界面 |
| `runtime.py` | 自动选择 CPU/NPU、加载模型和 NPU SDPA 适配 |
| `npu_patch.py` | 避免 `_transformer_encoder_layer_fwd` 回退 CPU 的等价实现 |

## API

```text
GET  /api/info
POST /api/snake/new    {"guarded": true, "prompt": "compact", "seed": 7}
POST /api/snake/step   {"session": "..."}
```

## 说明

`game.py` 和 `policy.py` 的规则、提示词与护栏逻辑参考自
[AXERA-TECH/laya-axera](https://github.com/AXERA-TECH/laya-axera) 的贪吃蛇示例，
推理接口适配为本仓库的 `laya` 包（CPU / Ascend NPU）。上游项目采用 Apache-2.0 许可。
