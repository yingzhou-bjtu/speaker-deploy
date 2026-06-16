# 板端内存与长期运行约束

## 结论（是否内存泄漏）

- **不是**典型 C 扩展泄漏：`librknnrt`/ONNX Runtime 单次 `InferenceSession` 常驻、每帧 `run()` 后临时数组由 Python GC 回收，属正常波动。
- **真正风险**是 **有界配置未打开时的无界增长** + **HDBSCAN 随缓冲变大的 CPU/峰值内存**，以及 **IPC/Redis 队列堆积**。

---

## 必须配置的约束（板端长期运行）

| 参数 | 建议值 | 作用 |
|------|--------|------|
| `max_hdbscan_utterances` | **60～200** | 缓冲超过 K 条后压缩。**不设则缓冲无限增长** |
| `max_store_utterances` | 与上相同或略大 | 写 npz 时上限（默认 10000，仅 `save_*` 时生效） |
| `buffer_compact_policy` | `fuse_only`（默认） | 只融合不删。融不动时缓冲可能 **暂时超过 K** |
| `fuse_min_intra_cluster_cosine` | 0.5～0.7 | 过低易误融，过高更难压回 K |
| IPC `jobs_maxsize` / `results_maxsize` | 64～256 | 背压。**消费端必须及时 `get` 结果** |
| Redis | 监控 `LLEN` | jobs/results 列表无人消费会 **无限涨** |
| `ort_threads` | 2～4（RK3588） | 过大与多 worker 争用 CPU/内存 |

**当前默认：** ``StreamingClusterPipeline`` 类构造默认不限 cap；**板端** ``start_robot.sh`` / ``run.py`` / ``api_server.py`` 经 ``pipeline_factory`` 统一为 **K=60**、``epsilon=0.05``。

---

## 内存大致量级（便于估 RSS）

| 组件 | 量级 |
|------|------|
| `eres2net_sv.onnx` + ORT | **~200～250 MB**（固定，进程启动后） |
| 嵌入缓冲 `_E_buf` | **K × 192 × 4 B ≈ 0.75×K KB**（如 K=60 ≈ 45 KB） |
| 并行列表 `_wav_keys` / `_stable_*` | 与 K、累计 mint 的 stable 数成正比 |
| **stable 注册表** `_stable_centroids` | 活跃条目仅 live 行；**FIFO 逻辑淘汰**不 shrink 内存数组，RSS 随 mint 单调增（见 §无界增长） |
| `_fusion_lineage` | 融合次数增加时增长 |
| 单次 `ingest` | 整段 wav + FBank + ONNX 中间张量（**随音频时长线性增**，结束后释放） |
| HDBSCAN 一步 | 与当前缓冲行数 n 相关，n 增大时 CPU 与峰值内存上升（建议 K≤200） |

---

## 无界增长路径（需避免）

1. **`max_hdbscan_utterances=None`（0 表示不设上限）**  
   每来一条 wav：缓冲 +1，并对 **全前缀** 做 HDBSCAN → 行数与耗时持续上升。

2. **stable_v_id 注册表**  
   每识别到新说话人 mint 新 ID；**聚类晋升**默认最多 **30** 条（`max_cluster_promoted_registry` / `SV_MAX_CLUSTER_PROMOTED_REGISTRY`），超出按 **FIFO** 淘汰最老 unpinned 条目（`_registry_evicted`，匹配/落盘不再包含）。**手动 `register` / 磁盘预注册**（pinned）不参与淘汰。  
   **内存注意**：淘汰只标记失效，**不释放** `_stable_centroids` 槽位；长期 7×24 仍需 `pipe.reset()` 或重启 worker。  
   **默认开启** promote（`enable_stable_cluster_promote=True`）：簇 size≥3 且 min intra cos≥0.75 时写代表向量。

3. **`fuse_only` 压不回 K**  
   找不到 cos≥阈值的同簇对时 compact 跳过，缓冲可 **长期 > K**（见 `skipped_compact`）。

4. **Redis worker**  
   `run.py worker` 单进程常驻一个 `StreamingClusterPipeline`。不 `shutdown`、不 `reset` 则状态一直累积。

5. **多进程**  
   每个 `spawn_streaming_worker` = **完整再加载一份 ONNX**（~200MB+ / 进程）。

---

## 运维策略

```python
# 长驻 worker 内（或定时任务）
if pipe.n_arrived > 500:  # 或按运行小时
    pipe.reset()
```

```bash
# 命令行显式限窗（可写入 systemd 或启动脚本）
python3 run.py audio --max-hdbscan-utterances 60 ...
python3 run.py worker --max-hdbscan-utterances 60 --ort-threads 4
```

- **单 worker** 处理全机器人声纹流。避免多 worker 重复占 200MB+ 模型内存。  
- 消费 Redis `speak:stream:results` 的速度 ≥ 生产速度。  
- 超长音频：上游切片（如 ≤30 s/条），降低单次 FBank 峰值。

---

## 环境变量（可选硬上限）

| 变量 | 含义 |
|------|------|
| `SV_MAX_HDBSCAN_UTTERANCES` | 覆盖默认 K（整数） |
| `SV_HARD_MAX_BUFFER` | 缓冲行数超过此值且无法 compact 时 **抛错**（防失控） |
| `SV_MAX_CLUSTER_PROMOTED_REGISTRY` | 聚类晋升注册表 FIFO 上限（默认 30；0=不限） |
| `SV_MAX_INGEST_AUDIO_SEC` | 单条 embed 最长秒数（默认 15；API/robot 共用） |
| `SV_PIPELINE_QUEUE_MAX` | `PipelineWorker` ingest 队列上限（默认 128，满则阻塞背压） |
| `SV_SPEAKER_REGISTRY` | 注册表 npz 路径（默认 `deploy/speaker_registry.npz`） |
| `SV_HDBSCAN_EPS` | HDBSCAN `cluster_selection_epsilon`（默认 0.05） |

---

## 不属于泄漏但会看到的现象

- `_E_buf` 扩容时旧 ndarray 等待 GC → RSS **阶梯上升**，通常稳定 plateau。  
- `multiprocessing.Queue` 满时生产者 **阻塞**，不是丢包也不是泄漏。  
- 用 `tracemalloc` / `/proc/<pid>/status` 的 VmRSS 观察：在 **固定 K + 定期 reset** 下应平稳。
