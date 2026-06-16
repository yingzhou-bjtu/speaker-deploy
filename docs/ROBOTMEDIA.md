# robotmedia 是什么？

**robotmedia** 是开发板上随机器人系统安装的多媒体/语音守护进程（`/usr/bin/robotmedia`，systemd 服务 `robotmedia.service`）。

它负责（与声纹 deploy **独立**）：

- 麦克风采集、VAD（语音活动检测）、唤醒词（如配置里的巴布巴布）
- 本地/云端 ASR、SRS 推流、表情 WebSocket 等
- 可选：把 **VAD 判定为说话** 的那段 PCM 通过 TCP 推给本机其它程序

## 和声纹服务的关系

| 组件 | 职责 |
|------|------|
| **robotmedia** | 硬件 + VAD + 何时推 PCM（不改它的代码） |
| **deploy/robot_service** | 连 TCP 收 PCM → ONNX 嵌入 → HDBSCAN → 输出 `v_id` |

对接配置（`config.ini`）：

```ini
[vad_pcm_socket]
enable = 1    # 必须为 1 才会在 :5001 推 PCM
```

协议：`PCM S16LE / 16kHz / mono`，**无头、无长度前缀**。一般在 **vad_start～vad_end** 期间推送，段结束常伴随 TCP 断开。

**不需要改 robotmedia 内部的 VAD 实现**。只需打开 `vad_pcm_socket`。

TCP 协议本身**没有 vad_end 帧**，只有裸 PCM。`pcm-vad-end-mode=silero_wav`（TCP 默认）时：

1. 5001 上持续缓冲 PCM（内存，不落盘）
2. 监听同目录新出现的 `silero_vad_*.wav`（robotmedia 在 `SILERO_VAD_END` 写入，与 `FeedPcm` 同段）
3. **见到 wav = vad_end** → `pcm-vad-end-audio=auto`（默认）：
   - 比对 TCP 缓冲帧数与 wav 元数据（`soundfile.info`，不读整文件）
   - **对齐** → 内存 `submit_audio`
   - **不齐** → 读落盘 wav

可选 `idle` / `silero_wav_or_idle` 作备选。

### wav_dir 与 TCP 的段界差异

| wav_dir | TCP |
|---------|-----|
| robotmedia 在 vad_end 写完整 wav，含 preroll | 旧实现曾在收流中按 idle 提前 submit，段更碎 |
| 段界与 ASR 一致 | 当前实现缓冲到段末再 analyze |

## 排障：5001 连上但没数据

实测（`scripts/probe_tcp_5001.py`）：

1. `enable=1` 且 `ss` 见 `robotmedia` 监听 `0.0.0.0:5001` → 配置正常。
2. **必须有一个客户端在说话前已连上 5001**。VAD 段内 robotmedia 才往该连接推 PCM（与落盘 `silero_vad_*.wav` 同段）。
3. 声纹服务与探测脚本不应同时连接 5001。排障时先 `pkill -f robot_service.py` 再跑 probe。
4. 板端默认 `pcm-source=wav_dir` 时**根本不会连 5001**，表现为 TCP 无数据。要内存流请 `SV_PCM_SOURCE=tcp`。

```bash
# 仅探测 5001（先停 robot_service）
pkill -f robot_service.py
.venv/bin/python scripts/probe_tcp_5001.py --duration 60

# 声纹走 TCP（不落盘 wav）
SV_PCM_SOURCE=tcp ./start_robot.sh
```
