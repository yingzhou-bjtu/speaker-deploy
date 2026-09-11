# HDBSCAN 流式折中算法：实验记录

## 1. 实验目的

验证标准 Full HDBSCAN、已有快速候选方法和风险感知自适应重聚类方法在连续嵌入流上的质量与效率差异。实验统一使用相同输入、相同 L2 归一化、相同 HDBSCAN 参数和相同检查点。

## 2. 可复现实验框架

实验目录：

`/home/user/Desktop/paper/hdbscan_streaming_bench`

数据接口统一为：

```text
X: float32 array, shape [N, D]
y: int array, shape [N]
```

真实音频流程为：

```text
FSDD WAV -> 16 kHz 重采样 -> FBank -> deploy ONNX 声纹模型 -> 192 维嵌入 -> L2 归一化 -> 流式聚类
```

当前固定数据为 FSDD 均衡子集，共 30 条音频、6 位说话人，每位 5 条。原始数据、抽样清单、嵌入和来源说明均保存在实验目录的 `data/fsdd/` 下。

## 3. 方法

| 方法 | 实验含义 |
|---|---|
| Full HDBSCAN | 每个新样本到达后，对完整历史前缀重新聚类 |
| 自适应折中方法 | 初始 Full 建立代表点缓存，稳定样本近邻分配，风险样本或周期预算触发 Full |
| FISHDBC | 已有增量 MST 方法 |
| AHC | 余弦距离平均链接凝聚层次聚类 |
| SC-pNA | ICASSP 2025 的自调节稀疏谱聚类嵌入级实现 |
| DIART-style | 参考 DIART 增量聚类阶段的 embedding-only 在线质心实现 |

自适应折中方法的默认配置为：`tradeoff_warmup=12`、`tradeoff_margin=0.02`、`tradeoff_max_distance=0.70`、`tradeoff_full_interval=4`、`min_cluster_size=3`、`min_samples=2`。

同步主方案候选 Adaptive-FISHDBC 维护一个稳定样本 buffer。样本先依据
最近一次 FISHDBC 簇代表点计算距离和间隔；高置信样本暂存，buffer 达到
`batch_size=4` 后一次提交并执行 FISHDBC cluster；低置信样本立即 flush，
从而优先保证边界样本的及时处理。`batch_size=1` 是退化一致性检查，应该
接近 FISHDBC；`batch_size>1` 才是计算量优化来源。

当前论文主方案候选为 Adaptive-FISHDBC-Async：它将 buffer flush 放到独立 worker。前台只
完成风险判断和 tentative label，worker 按输入顺序维护 FISHDBC；最终评估前
通过 `finalize()` 等待 worker 完成，因此质量指标仍来自完整提交后的正式标签。
异步版本的前台延迟与后台维护耗时必须分开报告，并同时记录 batch pending
和真实队列 `Queue.qsize()` 的最大值。

## 4. 评价指标

质量指标包括 Hungarian 映射 ACC、将噪声视为单独临时簇的 ACC、ARI、NMI、pairwise precision、pairwise recall、false merge rate 和 coverage。效率指标包括冷启动和稳态单步延迟、P95 延迟、总耗时、Python 峰值内存、Full 触发次数、缓存更新次数、不确定性次数和代表点距离比较次数。

其中，噪声单例 ACC 更接近当前 deploy 的公开标签口径，但不能替代严格 ACC。`cluster_accuracy_non_noise` 必须和 coverage 一起解释。对于说话人系统，false merge rate 是重要安全指标，不能只看 ACC。

## 5. 当前真实结果

数据：FSDD 30 条固定音频嵌入。基线和候选结果均来自实际运行产物。

| 方法 | 稳态平均单步延迟 ms | 相对 Full 加速 | 噪声单例 ACC | ARI | NMI | false merge rate | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full HDBSCAN | 54.91 | 1.00x | 0.767 | 0.661 | 0.805 | 0.013 | 0.767 |
| 自适应折中方法 | 5.35 | 10.25x | 0.767 | 0.661 | 0.805 | 0.013 | 0.767 |
| Adaptive-FISHDBC（batch=4） | 0.86 | 63.8x | 0.767 | 0.722 | 0.849 | 0.000 | 0.733 |
| FISHDBC | 1.06 | 51.9x | 0.767 | 0.722 | 0.849 | 0.000 | 0.733 |
| AHC | 1.30 | 42.34x | 0.733 | 0.680 | 0.850 | 0.008 | 1.000 |
| SC-pNA | 13.08 | 4.20x | 0.333 | 0.191 | 0.395 | 0.437 | 1.000 |
| DIART-style | 0.049 | 1116x | 0.833 | 0.664 | 0.844 | 0.061 | 1.000 |

相对 Full 的加速比使用稳态均值。Full 的首次调用、Python 导入和 HDBSCAN 初始化成本单独保留在原始 summary 中，因此表中的加速比不能解释为端到端总墙钟时间加速。

## 6. 自适应方法的计算量

默认保守配置的最终统计为：

- Full 触发 10 次，占 30 个输入步的 33.3%。
- 缓存更新 11 次，占 36.7%。
- 不确定性触发 6 次。
- 代表点距离比较 59 次，平均每个输入 1.97 次。
- 最终质量指标与 Full 完全一致。

这说明当前方法的速度收益来自减少 Full 重算，但由于数据规模很小，整体墙钟时间仍会受到 ONNX/HDBSCAN 初始化和 Python 环境成本影响。论文实验必须在更长序列上报告稳态延迟和总计算量。

## 7. 参数 sweep 结论

参数表保存在：

`results/fsdd_tradeoff_sweep/tradeoff_sweep.csv`

在当前数据上，`max_distance=0.70` 的多个周期设置都保持 Full 的最终质量。更宽松的距离阈值和更长的 Full 周期可以减少触发，但可能出现质量下降。例如 `max_distance=0.85, full_interval=16` 的噪声单例 ACC 降至 0.567，false merge rate 增至 0.139。这个现象支持风险触发机制的必要性，但还不能作为跨数据集规律。

本轮核心基线结果表明：AHC 是很强的经典速度基线；SC-pNA 的小样本表现不稳定，不能仅凭本 smoke 否定其原论文结果；DIART-style 是 embedding-only 工程代理，速度很高但 false merge rate 达到 0.061，不适合作为安全约束下的最终方案。所有数值来自当前 `compare_to_full.csv`，不代表对应论文在其标准数据集上的原始报告结果。

异步验证使用独立 workload：固定样本到达间隔 `5 ms`，结果保存在
`results/async_validation/async_comparison.csv`。FSDD 上 Adaptive-FISHDBC-Async
前台稳态 P95 为 `0.471 ms`，FISHDBC 为 `4.036 ms`，最终 ACC 为 `0.767`、
ARI 为 `0.722`、false merge rate 为 `0`。FSDD 前台稳态均值为 `0.183 ms`，
后台维护总耗时约 `65 ms`，真实队列最大深度为 `1`。320 条合成流上前台稳态 P95 为
`0.825 ms`，FISHDBC 为 `8.766 ms`；两者最终 ACC `0.916`、ARI `0.878`、
false merge rate `0`。后台维护总耗时约 `1280 ms`，真实队列最大深度为 `2`，
batch pending 最大值为 `4`。该结果证明异步方案改善了决策尾延迟，但不能省略
后台资源成本和队列积压约束。

异步队列采用有界容量，默认 `tradeoff_max_queue=8`，并记录
`max_queue_depth`、`backpressure_count` 和 `backpressure_ms_total`。在极快输入、
队列上限设为 `2` 的过载测试中，真实队列深度严格为 `2`，触发背压 `299` 次，
最终 ACC `0.916`、ARI `0.878`、false merge rate `0`；前台稳态均值退化为
`5.07 ms`。这说明方案在负载过高时选择显式背压，而不是无界积压或静默丢弃。

额外的退化一致性检查在 320 条合成流上设置 `batch_size=1`：Adaptive-FISHDBC-
Async 的最终 ACC `0.916`、ARI `0.878`、false merge rate `0`，与 FISHDBC 一致；
前台稳态均值为 `0.468 ms`，最大积压为 `1`。因此 `batch_size=1` 可作为实现
正确性检查，`batch_size=4` 才是主方案的计算优化配置。

新增漂移/晚加入场景：共 8 位说话人、320 条嵌入，前 160 步只出现前 4 位，
后 160 步加入其余 4 位，同时每位说话人的中心逐步漂移。Adaptive-FISHDBC-
Async 在 `5 ms` 到达间隔下前台稳态 P95 为 `0.684 ms`，FISHDBC 为 `8.892 ms`；
最终 ACC、ARI、NMI 均为 `1.0`，false merge rate 为 `0`，最大队列积压为 `4`。
该结果只说明受控漂移场景下没有观察到质量退化，不能替代真实长时说话人数据。

到达速率 sweep 结果保存在 `results/rate_sweep_v2/rate_sweep.csv`。在 `1/5/10/20 ms`
四种到达间隔下，Adaptive-FISHDBC-Async 的最终 ACC、ARI、NMI 和 false merge
均与 FISHDBC 一致，batch pending 最大值为 `4`，真实队列最大深度为
`3/2/2/2`；前台决策 P95 分别约为 `2.01/0.79/1.28/1.71 ms`，FISHDBC
分别约为 `8.92/10.33/11.88/10.56 ms`。
后台维护耗时随负载和 worker 调度变化，必须与前台延迟分开报告。

多 seed 验证使用 `20260910/20260911/20260912` 三个 seed、8 位说话人和
`5 ms` 到达间隔。Adaptive-FISHDBC-Async 与 FISHDBC 的 ACC、ARI、NMI 和
false merge rate 在三个 seed 上逐一一致；前台均值分别为 `0.407 ms` 和
`5.795 ms`，前台 P95 均值分别为 `0.753 ms` 和 `9.452 ms`，约为
`14.2x` 和 `12.5x` 的差异。真实队列最大深度均为 `2`。原始结果在
`results/multiseed_validation/multiseed.csv`，统计结果在
`multiseed_summary.json`。

在 320 条合成流上，Adaptive-FISHDBC（batch=4）与 FISHDBC 的最终 ACC、ARI、NMI 和 false merge rate 一致，稳态均值由 `4.99 ms` 降至 `3.06 ms`，约 `1.63x` 加速；FISHDBC 的刷新次数为 `320`，Adaptive-FISHDBC 为 `90`。这说明批量提交机制在较长序列上有计算收益，但仍需真实多说话人数据验证。

需要同时报告尾延迟：FSDD smoke 上 Adaptive-FISHDBC 的 P95 为 `4.01 ms`，FISHDBC 为 `1.60 ms`；320 条合成流上分别为 `16.25 ms` 和 `8.99 ms`。因此当前版本是平均延迟和刷新次数更优、尾延迟更差的折中，不应声称全面优于 FISHDBC。

## 8. 图表和复现命令

结果对比表（核心 5 基线 + 本文方法）：

`results/fsdd_optimization_candidates/compare_to_full.csv`

速度-质量-动作图：

`results/fsdd_optimization_candidates/tradeoff_report.png`

完整运行：

```bash
cd /home/user/Desktop/paper/hdbscan_streaming_bench
bash scripts/run_tradeoff_report.sh
```

音频 smoke 会复用经过 manifest 和模型哈希校验的嵌入缓存，避免重复执行声纹
模型推理。需要验证模型变更时，应使用 `audio_to_embeddings.py --force` 强制
重算。日常运行默认跳过 `fast_hdbscan` 静态控制项，完整控制实验可设置
`RUN_FAST_HDBSCAN=1`。

本次工程优化实测：首次生成缓存时单条嵌入推理均值约 `115.9 ms`，同配置的
第二次入口命中缓存，进程墙钟时间约 `0.68 s`。该收益属于实验准备阶段的
嵌入复用，不计入聚类方法的稳态加速比，也不改变 `X`、`y` 或任何质量指标。

参数 sweep：

```bash
.venv/bin/python scripts/run_tradeoff_sweep.py \
  --data data/fsdd/embeddings/fsdd_smoke.npz \
  --full results/fsdd_full_baseline/summary.json \
  --out results/fsdd_tradeoff_sweep
```

## 9. 证据边界

Full benchmark 与 deploy 侧稠密簇标签一致性验证通过。当前结果只证明本地 CPU 环境、小规模 FSDD 子集和现有声纹模型上的工程行为。FSDD 是英文数字语音，而现有模型是中文声纹模型，因此该数据不适合直接支撑跨语言声纹识别质量结论。

当前没有完成 MUSA GPU 运行，也没有完成大规模真实说话人数据、真实端到端
音频延迟和注册身份匹配实验。上述内容不能在论文中写成已验证结果；受控
长流压力测试已经完成，但不等价于真实部署的长期运行。

当前已加入 640 条受控漂移流的长流入口 `scripts/run_long_stress.sh`，并将
`peak_rss_mb` 纳入 benchmark。长流测试完成后，必须同时检查最终 evaluation
points、真实队列深度、背压次数和 RSS；仅有平均延迟不足以证明系统可长期运行。

本次 640 条长流实测最终 evaluation points 为 `640/640`，ACC `0.9969`、
ARI `0.9986`、false merge rate `0`，后台维护 `215` 次，真实队列最大深度
`6`，峰值 RSS `123.9 MB`，未触发背压。该结果支持受控长流下的完整性和
内存边界，但仍需真实音频长时间测试。

## 10. 下一步实验顺序

1. 接入与声纹模型语言和场景匹配的真实说话人数据，并固定训练集、验证集和测试集划分。
2. 在相同数据顺序下完成 Full、FISHDBC 和自适应折中方法的多次随机种子实验。
3. 增加说话人加入、说话人漂移、噪声和簇规模变化场景。
4. 加入簇内离散度、HDBSCAN probability、新簇候选检测和风险校准。
5. 最后再选择 MUSA 设备，单独测量 CPU、MUSA、端到端音频和长期运行结果。

## 11. 近期论文基线来源

- SC-pNA：Raghav et al., “Self-Tuning Spectral Clustering for Speaker Diarization,” ICASSP 2025, DOI `10.1109/ICASSP49660.2025.10890194`，论文预印本见 <https://arxiv.org/abs/2410.00023>。
- DIART：Coria et al., “Diart: A Python Library for Real-Time Speaker Diarization,” JOSS 2024，代码见 <https://github.com/juanmc2005/diart>。本实验只复现其 embedding-level 增量聚类思想，不引入 pyannote 分割模型，因此标为 `diart_style`，不能写成完整 DIART 复现。
- O-EENC-SD：ICASSP 2025，代码见 <https://github.com/egruttadauria98/O-EENC-SD>。它是 EEND-EDA 端到端在线模型，需要独立的分割和 attractor 网络，暂不混入当前 embedding-only 核心表。
- Streaming Sortformer：INTERSPEECH 2025，论文见 <https://arxiv.org/abs/2507.18446>。它依赖 Sortformer 模型和 Arrival-Order Speaker Cache，暂不混入当前 HDBSCAN 输入契约。
