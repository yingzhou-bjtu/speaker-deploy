# Speaker Deploy · 声纹识别板端系统

面向 **RK3588 等嵌入式板端** 的流式说话人识别（声纹）部署包：ONNX CPU 推理 + 在线 HDBSCAN 聚类 + HTTP/TCP 业务接入。适用于机器人语音链路（robotmedia VAD PCM）与离线 wav 批处理场景。

| 项目 | 说明 |
|------|------|
| 模型 | ERes2NetV2 声纹嵌入（`iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common`） |
| 推理 | ONNX Runtime（CPU），**不含 RKNN** |
| 聚类 | 流式 HDBSCAN，自动 mint `v_id` |
| 注册 | 预注册 + 无监督簇晋升，共享 `speaker_registry.npz` |
| 典型平台 | RK3588，4 路 ONNX 线程 |

---

## 功能概览

- **板端一体服务**：对接 robotmedia `[vad_pcm_socket] :5001`，PCM → 嵌入 → 聚类 → `v_id`，同进程提供 HTTP API
- **HTTP API**：健康检查、聚类面板、按 wav 路径 ingest
- **常驻 serve**：stdin / 目录扫描 / Redis 队列多种输入模式
- **声纹注册**：`register` 子命令写入注册表，cos≥0.85 命中时跳过聚类
- **探测脚本**：板端瓶颈分析、TCP/PCM 冒烟、preflight 检查

---

## 目录结构

```
├── speaker_id/          # 核心：嵌入、聚类、流式管线、ONNX 后端
├── pretrained/          # eres2net_sv.onnx（Git LFS）
├── scripts/             # 探测与冒烟脚本
├── docs/                # robotmedia 协议说明
├── run.py               # 统一 CLI 入口
├── robot_service.py     # 板端 robotmedia + API 一体服务
├── api_server.py        # 独立 HTTP API
├── start_robot.sh       # 板端推荐启动脚本
├── start_api.sh
├── requirements.txt
├── CONSTRAINTS.md       # 内存与长期运行约束（必读）
└── 板端启动.md          # 板端部署步骤
```

---

## 快速开始

### 1. 安装依赖

```bash
./setup_board_deps.sh
# 或手动：
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 安装顺序见 `setup_board_deps.sh`；**不要单独** `pip install hdbscan`，否则可能把 numpy 升到 2.x 导致不兼容。

### 2. 板端对接 robotmedia（推荐）

```bash
source .venv/bin/activate
# 需要 ROS 时间戳时：source /opt/ros/humble/setup.bash

export SV_PCM_SOURCE=tcp
./start_robot.sh
```

等价于 `python run.py serve-robot --pcm-source tcp`。启动前会自动执行 `scripts/preflight_board.py`（检查 :5001 等）。

- 默认 `pcm-mode vad`：一段 VAD → 一次 `ingest`
- 调试：`SV_VERBOSE=1`
- TCP 不可用：不设 `SV_PCM_SOURCE` 或 `SV_PCM_SOURCE=wav_dir`，扫描落盘 wav

协议与排障详见 [docs/ROBOTMEDIA.md](docs/ROBOTMEDIA.md)、[板端启动.md](板端启动.md)。

### 3. 仅 HTTP API

```bash
source .venv/bin/activate
python api_server.py --port 8765 --ort-threads 4 --max-hdbscan-utterances 60
```

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/health` | 存活探测 |
| GET | `/v1/panel` | 各 `v_id` 条数、不确定条数 |
| POST | `/v1/ingest` | JSON：`{"wav_path": "/abs/path.wav", "wav_key": "optional"}` |

### 4. PC 离线常驻

```bash
source .venv/bin/activate
export SV_ORT_THREADS=4
export SV_MAX_HDBSCAN_UTTERANCES=60

python run.py serve --verbose --ort-threads 4 --max-hdbscan-utterances 60
```

stdin 每行一个 wav 绝对路径，输入 `quit` 退出；输出 `文件名 -> v_id` 或 `?`。

---

## 声纹注册

预注册与无监督晋升共用 **`speaker_registry.npz`**，服务启动时自动加载。

```bash
python run.py register alice /path/to/reg.wav
```

流式路径：**embed → 匹配注册表（cos≥0.85 跳过聚类）→ 未命中则 HDBSCAN → 稳定簇晋升注册表**。

---

## 关键参数

| 参数 / 环境变量 | 默认 | 说明 |
|-----------------|------|------|
| `SV_PCM_SOURCE` / `--pcm-source` | `wav_dir` | 板端设 `tcp`，连 robotmedia :5001 |
| `--ort-threads` / `SV_ORT_THREADS` | 4 | ONNX 线程数 |
| `--max-hdbscan-utterances` / `SV_MAX_HDBSCAN_UTTERANCES` | 60 | 流式缓冲上限，`0` 表示全前缀 |
| `--hdbscan-min-cluster-size` | 3 | HDBSCAN 最小簇大小 |

长期 7×24 运行请务必阅读 [CONSTRAINTS.md](CONSTRAINTS.md)。

---

## 探测与冒烟

```bash
# 挂载 / 流式 ingest 耗时与 RSS
python run.py probe --audio-dir /data/local/tmp/v_zhouying -n 20 \
  --ort-threads 4 --max-hdbscan-utterances 60 \
  --out /tmp/sv_probe.json

python3 scripts/probe_board_bottleneck.py --audio-dir /data/local/tmp/v_zhouying -n 20
SMOKE_N=5 python3 scripts/smoke_stream_实际录音.py
```

---

## 模型文件

| 文件 | 说明 |
|------|------|
| `pretrained/eres2net_sv.onnx` | 声纹嵌入 ONNX（~205MB，Git LFS） |
| `pretrained/model_reference.json` | 模型元数据 |

---

## 许可证与隐私

本仓库为 **私有** 项目，内含预训练模型与业务部署脚本，请勿对外分发模型权重。

---

## 相关文档

- [docs/ROBOTMEDIA.md](docs/ROBOTMEDIA.md) — TCP/PCM 协议
- [板端启动.md](板端启动.md) — RK3588 部署清单
- [CONSTRAINTS.md](CONSTRAINTS.md) — 内存与无界增长约束
