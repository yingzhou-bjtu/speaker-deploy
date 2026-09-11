# 风险感知异步增量聚类：理论框架与形式化证明

## 1. 论文要回答的问题

本文不主张重新发明 HDBSCAN，而是研究一个流式说话人识别中的选择性计算问题：

> 在持续到达的说话人嵌入流上，如何让稳定样本以低延迟完成分配，同时把昂贵的密度聚类维护集中给真正不确定的样本，并优先抑制跨说话人误合并？

因此，`Adaptive-FISHDBC-Async` 的核心不是“所有样本都增量更新”，而是一个**风险感知的异步选择性重聚类框架**：

```text
嵌入到达
  -> 提取距离、margin、簇内离散度、老化和新簇证据
  -> 稳定样本走 FAST 快路径
  -> 不确定样本走 REFRESH 刷新路径
  -> 后台 FISHDBC 维护，提交前等待相关状态完成
```

全文理论链固定为：

```text
问题定义
  -> 几何假设
  -> 距离与 margin 引理
  -> 快路径可接受性推论
  -> 局部等价与风险完备假设
  -> 逐前缀一致性主定理
  -> 队列有界与代价分解
  -> 受约束的延迟-风险优化
```

这条链中，几何分析负责解释“为什么可以快速处理一部分样本”；控制定理负责解释“为什么风险样本必须刷新以及刷新后如何保证提交语义”；实验负责验证这些假设在真实声纹流上是否成立。

## 2. 符号、参考对象和输出语义

设归一化说话人嵌入流为：

```math
X_T=(x_1,x_2,\ldots,x_T), \qquad x_t\in\mathbb{R}^d,
\qquad \|x_t\|_2=1.
```

定义参考聚类器 `R`：

```math
R_t = R(X_t).
```

`R` 必须在论文中固定。若主实验把 FISHDBC 作为在线参考，就在理论中统一写作 `R=FISHDBC`；Full HDBSCAN 可以作为外部质量基准，但不能在同一个定理里交替充当参考对象。

本文讨论的是**提交时输出**。设 `\hat y_t` 是系统在时间 `t` 允许提交的标签，而不是后台尚未收敛时的临时标签。系统目标是：

```math
\hat y_t = R_t
```

或在不要求逐标签完全一致时，至少满足受约束的误合并风险目标。这个区分很重要：后台任务完成前的 tentative label 不能直接拿来宣称最终识别结果。

对活动簇 `k`，系统维护压缩状态：

```math
r_{k,t}=normalize\left(\sum_{i\in C_{k,t}}x_i\right),
```

```math
s_{k,t}=\frac{1}{|C_{k,t}|}\sum_{i\in C_{k,t}}\|x_i-r_{k,t}\|_2.
```

其中 `r_{k,t}` 是快路径代表点，`s_{k,t}` 是簇内离散度。它们是用于决策的缓存状态，不等价于 HDBSCAN 的完整互可达距离图或密度树。

理论距离采用 Euclidean 距离：

```math
d_2(x,y)=\|x-y\|_2.
```

实现中可以使用 cosine distance，因为单位向量满足：

```math
\|x-y\|_2^2=2(1-\langle x,y\rangle).
```

因此最近代表点的排序一致；但涉及三角不等式的证明必须使用 `d_2`，不能把 `1-\langle x,y\rangle` 在一般情形下直接称为度量。

## 3. 风险门控与两条执行路径

令最近和次近代表点距离为：

```math
d_{1,t}=\min_k d_2(x_t,r_{k,t}),
```

```math
d_{2,t}=\min_{j\ne k_t}d_2(x_t,r_{j,t}),
\qquad
m_t=d_{2,t}-d_{1,t}.
```

其中 `k_t` 是最近代表点对应的簇，`m_t` 是最近和次近候选之间的 margin。另有：

- `s_{k_t,t}`：最近簇的离散度；
- `a_t`：自上次刷新以来的年龄；
- `n_t`：新簇或未知说话人证据。

硬门控定义为：

```math
g_t=1\iff
d_{1,t}\le\delta
\land m_t\ge\mu
\land s_{k_t,t}\le\sigma_{max}
\land a_t<H
\land n_t=0.
```

输出路径为：

```math
\hat y_t=
\begin{cases}
F(X_t), & g_t=1,\\
G(X_t), & g_t=0.
\end{cases}
```

`F` 是低成本代表点分配，`G` 是 FISHDBC 刷新或等价的风险处理路径。连续风险分数可用于校准：

```math
\mathcal{R}_t=
w_d[d_{1,t}/\delta]_+
w_m[(\mu-m_t)/\mu]_+
w_s[s_{k_t,t}/\sigma_{max}]_+
w_a a_t/H
w_n n_t.
```

论文主逻辑建议使用硬门控，连续分数用于阈值扫描、可视化和消融实验。

## 4. 假设体系

为了避免把实验现象伪装成定理，假设分成三类。

### 假设 G：几何可分性

若样本 `x_t` 属于真实簇 `k`，另一个说话人为 `j`，存在真实中心 `c_k,c_j`，并满足：

```math
d_2(x_t,c_k)\le\eta,
```

```math
d_2(c_k,r_{k,t})\le\rho,
\qquad
d_2(c_j,r_{j,t})\le\rho,
```

```math
d_2(c_k,c_j)\ge\Delta.
```

`eta` 表示局部嵌入漂移，`rho` 表示代表点误差，`Delta` 表示跨说话人中心间隔。

### 假设 L：局部参考等价

当几何证据满足快路径门控、且没有新簇、老化或高离散度信号时：

```math
F(X_t)=R(X_t).
```

这是从“代表点近”到“参考聚类器给出相同标签”的桥接假设。几何分离本身不能推出 HDBSCAN 的密度连通性等价，因此该假设必须通过实验或额外模型分析支持。

### 假设 C：风险完备和刷新最终性

风险完备性要求所有可能导致跨说话人误合并的状态都满足以下至少一项：

```math
g_t=0
```

或在最多 `H` 个样本后强制刷新。

刷新最终性要求：最终提交 `\hat y_t` 之前，所有会影响该输出的刷新任务已经完成，并且：

```math
G(X_t)=R(X_t).
```

### 假设 S：后台服务能力

令刷新任务平均到达率为 `lambda_arrival`，后台服务率为 `mu_worker`。长期低积压需要：

```math
lambda_arrival < mu_worker.
```

这不是正确性假设，而是系统性能假设。队列容量只能限制资源占用，不能保证没有背压。

## 5. 几何引理：从误差界到 margin 下界

### 引理 1：同簇代表点距离上界

在假设 G 下：

```math
d_2(x_t,r_{k,t})\le\eta+\rho.
```

**证明。** 由三角不等式：

```math
d_2(x_t,r_{k,t})
\le d_2(x_t,c_k)+d_2(c_k,r_{k,t})
\le \eta+\rho.
```

### 引理 2：异簇代表点距离下界

在假设 G 下：

```math
d_2(x_t,r_{j,t})\ge\Delta-\eta-\rho.
```

**证明。** 由三角不等式的逆形式：

```math
d_2(c_k,c_j)
\le d_2(c_k,x_t)+d_2(x_t,r_{j,t})+d_2(r_{j,t},c_j).
```

结合假设 G 得：

```math
\Delta\le\eta+d_2(x_t,r_{j,t})+\rho,
```

移项即得结论。

### 引理 3：margin 下界

若 `k` 是真实簇，且所有竞争簇代表点满足假设 G，则：

```math
m_t
=d_2(x_t,r_{j,t})-d_2(x_t,r_{k,t})
\ge\Delta-2(\eta+\rho).
```

**证明。** 将引理 1 的上界和引理 2 的下界相减：

```math
m_t
\ge [\Delta-\eta-\rho]-[\eta+\rho]
=\Delta-2(\eta+\rho).
```

### 推论 1：几何可接受快路径

若：

```math
\delta\ge\eta+\rho,
\qquad
\mu\le\Delta-2(\eta+\rho),
```

并且：

```math
s_{k_t,t}\le\sigma_{max},\qquad a_t<H,\qquad n_t=0,
```

则该样本满足 `g_t=1`。若：

```math
\Delta>2(\eta+\rho),
```

则最近代表点与竞争代表点之间存在严格正 margin。

**边界说明。** 推论 1 只说明代表点选择在几何意义上稳定，不说明 `F` 与 HDBSCAN 或 FISHDBC 必然一致。这个缺口由假设 L 明确承接，而不是被隐藏在“安全”一词里。因此论文正文中更建议称其为 **geometrically admissible fast path**，中文称“几何可接受快路径”。

## 6. 主定理：逐前缀参考一致性

### 定义 1：提交前缀

称前缀 `X_t` 已提交，当且仅当系统已经等待完所有影响其输出的刷新任务。只有提交前缀上的 `\hat y_t` 才进入最终指标。

### 定理 1：条件式逐前缀一致性

若假设 L、C 成立，且所有输出都只在提交前缀上产生，则对任意 `t`：

```math
\hat y_t=R(X_t).
```

**证明。** 对 `t` 做归纳。

**基例。** `t=1` 时，若 `g_1=1`，由假设 L 有 `F(X_1)=R(X_1)`；若 `g_1=0`，由刷新最终性和刷新契约有 `G(X_1)=R(X_1)`。因此 `\hat y_1=R(X_1)`。

**归纳步。** 假设前 `t-1` 个已经提交的前缀满足结论。处理 `X_t`：

- 若 `g_t=1`，由局部参考等价，`\hat y_t=F(X_t)=R(X_t)`；
- 若 `g_t=0`，系统等待相关刷新完成，由刷新最终性，`\hat y_t=G(X_t)=R(X_t)`。

两种情况都成立，因此结论对 `t` 成立。由数学归纳法，所有提交前缀均满足一致性。

### 推论 2：误合并风险的传递

若参考聚类器 `R` 在提交前缀上不存在跨说话人误合并，则本文系统也不存在跨说话人误合并。

**证明。** 由定理 1，本文输出和 `R` 输出逐前缀相同；将参考输出的无误合并性质代入即可。

这里必须强调：推论 2 是参考一致性的蕴含关系，不是无条件的说话人识别准确率保证。若参考聚类器本身误合并，本文不会因为形式化证明而自动消除该错误。

## 7. 从几何稳定性到可验证风险上界

前面的几何推导只给出了“最近代表点不会轻易被竞争簇超过”的充分条件。要把它变成算法决策依据，还需要把几何条件写成一个可计算的风险证书。定义几何证书裕量：

```math
q_t=\Delta_t-2(\eta_t+\rho_t),
```

其中 `\Delta_t` 是当前样本与最危险竞争簇之间的估计中心间隔，`\eta_t` 是局部漂移上界，`\rho_t` 是代表点误差上界。令 `b_t` 为由校准集估计的参考不一致风险上界，并要求：

```math
q_t\ge 0 \quad\Longrightarrow\quad
P(F(X_t)\ne R(X_t)\mid z_t)\le b_t,
```

其中 `z_t=(d_{1,t},m_t,s_{k_t,t},a_t,n_t)` 是门控可观测状态。这里的 implication 是模型假设与校准程序的接口，而不是仅由欧氏距离自动推出的事实。可采用保守的分段上界：

```math
b_t=b_{geom}(q_t)+b_{drift}(s_{k_t,t},a_t)+b_{novel}(n_t),
```

并令每一项在几何裕量增大、簇内离散度减小、状态更新较新以及新簇证据消失时不增。于是，快路径的“可接受”不再只是一个经验阈值，而是一个可审计条件：

```math
\mathcal{A}_\varepsilon
=\{z_t:b_t\le\varepsilon_t\}.
```

### 定理 2：快路径的条件风险保证

若假设 G 给出 `q_t\ge0`，假设 L 保证在该局部区域内 `F(X_t)=R(X_t)`，且 `b_t` 是上述参考不一致事件的有效上界，则对所有满足 `z_t\in\mathcal{A}_\varepsilon` 的快路径样本：

```math
P(F(X_t)\ne R(X_t)\mid z_t)\le\varepsilon_t.
```

**证明。** 由 `z_t\in\mathcal{A}_\varepsilon` 的定义有 `b_t\le\varepsilon_t`。由风险上界的定义，`P(F(X_t)\ne R(X_t)\mid z_t)\le b_t`。合并两式即得结论。若进一步采用假设 L 的确定性局部等价，则该条件概率为 `0`，定理退化为逐样本一致性结论。

这个定理把论文中的三个层次区分开：几何分析产生 `q_t`，校准过程产生 `b_t`，门控策略只接受 `b_t` 不超过预算的状态。这样，实验可以直接画出 `b_t` 或其校准分位数与真实 FMR 的关系，而不是只报告一个未经解释的 accuracy。

## 8. Tradeoff 最优性：在风险约束下尽可能多走快路径

### 定义 2：可证明策略类

考虑只在 `FAST` 和 `REFRESH` 之间选择的策略类 `\Pi_\varepsilon`。策略 `\pi\in\Pi_\varepsilon` 必须满足：当它选择 `FAST` 时，当前状态属于 `\mathcal{A}_\varepsilon`；当状态不属于 `\mathcal{A}_\varepsilon` 时只能选择 `REFRESH`。这一定义把“正确性约束”放在策略类中，而不是事后用平均 accuracy 替代风险约束。

设单样本前台代价满足：

```math
c_f<c_r,
```

其中 `c_f` 是代表点快路径代价，`c_r` 是等待或触发刷新路径的等效代价。定义策略的快路径覆盖率：

```math
\kappa(\pi)=P(\pi(z)=FAST),
```

平均计算代价为：

```math
J(\pi)=c_f\kappa(\pi)+c_r(1-\kappa(\pi))
=c_r-(c_r-c_f)\kappa(\pi).
```

### 定理 3：风险约束下的点态最优门控

定义最大可行门控策略：

```math
\pi^*(z)=
\begin{cases}
FAST, & z\in\mathcal{A}_\varepsilon,\\
REFRESH, & z\notin\mathcal{A}_\varepsilon.
\end{cases}
```

在 `c_f<c_r` 且 `\Pi_\varepsilon` 的定义成立时，对任意 `\pi\in\Pi_\varepsilon`：

```math
\kappa(\pi)\le\kappa(\pi^*),
\qquad
J(\pi^*)\le J(\pi).
```

因此，`\pi^*` 在该策略类中同时最大化快路径覆盖率并最小化平均计算代价。它不是无条件地优于所有可能算法，而是在“风险不超过 `\varepsilon`、只能使用快路径或刷新路径、风险证书有效”这三个明确条件下的最优策略。

**证明。** 对任意状态 `z`：若 `z\notin\mathcal{A}_\varepsilon`，可行性要求所有 `\pi\in\Pi_\varepsilon` 都选择 `REFRESH`，与 `\pi^*` 相同；若 `z\in\mathcal{A}_\varepsilon`，`\pi^*` 选择 `FAST`，其他可行策略至多选择 `FAST`，因此逐状态有：

```math
I[\pi(z)=FAST]\le I[\pi^*(z)=FAST].
```

对状态分布取期望得到 `\kappa(\pi)\le\kappa(\pi^*)`。代入 `J(\pi)=c_r-(c_r-c_f)\kappa(\pi)`，并利用 `c_r-c_f>0`，得到 `J(\pi^*)\le J(\pi)`。

### 推论 3：显式的风险-代价 Pareto 前沿

当风险预算 `\varepsilon` 增大时，集合 `\mathcal{A}_\varepsilon` 单调扩大，因此：

```math
\varepsilon_1\le\varepsilon_2
\Longrightarrow
\kappa(\pi^*_{\varepsilon_1})\le
\kappa(\pi^*_{\varepsilon_2}),
```

且：

```math
J(\pi^*_{\varepsilon_2})\le
J(\pi^*_{\varepsilon_1}).
```

这给出了论文所需的 tradeoff 方向：放宽可接受风险预算不会降低快路径覆盖率，也不会增加理想化平均计算代价；收紧风险预算则必然牺牲一部分覆盖率或引入更多刷新。实际系统还受到队列和背压影响，因此最终实验应报告 `FMR`、`coverage`、`P95` 和 `refresh ratio` 的完整 Pareto 曲线，而不是只挑一个 operating point。

### 推论 4：固定覆盖目标下的最小风险选择

如果实验还要求 `\kappa(\pi)\ge\gamma`，则应在满足覆盖率的策略中选择最小风险的状态集合，而不是简单提高阈值。一个可操作的版本是令样本按 `b_t` 从小到大排序，优先将风险证书最强的样本交给 `FAST`，直到达到 `\gamma`。这相当于求解：

```math
\min_{A}\quad E[b_t\mid z_t\in A]
\qquad
\text{s.t.}\quad P(z_t\in A)\ge\gamma.
```

**证明。** 设可行集合 `A` 中包含状态 `z_h`，但集合外存在 `z_l`，且 `b(z_l)<b(z_h)`。将 `z_h` 替换为 `z_l` 不改变集合的测度，因此仍满足 `P(z_t\in A)\ge\gamma`；同时集合内风险总和严格下降，集合测度不变，所以条件平均风险严格下降。反复执行该交换，直到集合包含所有更低风险状态而不包含更高风险状态，所得集合必为风险阈值集合 `A_\tau=\{z:b(z)\le\tau\}`，必要时在边界处随机取样以精确满足覆盖率。故风险证书排序是固定覆盖目标下的最小经验风险选择。这一结果解释了为什么连续风险分数应参与阈值扫描，即使线上实现仍采用硬门控。

需要区分两个结论：定理 3 证明的是固定风险预算下的最低代价，推论 4 证明的是固定快路径覆盖率下的最低证书风险。二者合起来才构成本文的 tradeoff 理论，而不是单独声称“Adaptive 一定优于 Full”。

## 9. 资源和延迟命题

令后台队列容量为 `B`，任务到达数和完成数分别为 `u_t,v_t`：

```math
q_{t+1}=min\{B,\;max\{0,\;q_t+u_t-v_t\}\}.
```

### 命题 1：队列有界

若 `q_0\in[0,B]`，则对任意 `t`：

```math
0\le q_t\le B.
```

**证明。** `max{0,\cdot}` 保证下界，`min{B,\cdot}` 保证上界；对每个时间步递推即可。Tau Ceti 对该截断更新以及单任务入队特例进行形式化检查。

### 命题 2：前台延迟分解

若门控、暂定标签和入队的成本分别为 `c_g,c_l,c_q`，背压等待时间为 `w_t`，则：

```math
L_t^{front}=c_g+c_l+c_q+w_t.
```

这不是算法正确性定理，而是用于实验归因的可测量分解。

### 命题 3：平均计算成本

设快路径成本为 `c_f`，一次后台维护成本为 `c_m(n)`，风险刷新比例为 `p_r`，则：

```math
E[C/T]\approx c_f+p_r\,c_m(n).
```

该式揭示方法收益来源：只有当 `p_r` 足够小，且后台维护不阻塞前台，选择性计算才会优于每样本完整刷新。

## 10. 统一优化目标

令：

```math
\theta=(\delta,\mu,\sigma_{max},H,B,b),
```

其中 `b` 是后台批大小。定义跨说话人误合并率：

```math
FMR=
\frac{\sum_{i<j}I[y_i\ne y_j]I[\hat y_i=\hat y_j]I[\hat y_i\ne noise]I[\hat y_j\ne noise]}
{\sum_{i<j}I[y_i\ne y_j]}.
```

参数选择写成：

```math
\min_{\theta}\quad
\mathbb{E}[L_t^{front}]
\,+\,\lambda_r\mathbb{E}[N_{refresh}/T]
\,+\,\lambda_c\mathbb{E}[C_{refresh}/T]
```

约束为：

```math
FMR(\theta)\le\epsilon,
\qquad
coverage(\theta)\ge\gamma,
\qquad
Q_{max}(\theta)\le B.
```

这正面呼应论文核心目标：在误合并风险受控、有效覆盖率不低于目标值的前提下，减少前台延迟和重复密度聚类计算。

## 11. 论文中的理论贡献表述

建议正文写成：

1. 提出一个风险感知异步选择性聚类框架，将稳定嵌入送入低成本快路径，将不确定嵌入送入 FISHDBC 刷新路径。
2. 建立基于样本漂移、代表点误差和簇间隔的 margin 分析，给出快路径门控阈值的可解释条件。
3. 在局部参考等价、风险完备和刷新最终性假设下，证明提交前缀与参考聚类器一致，并给出队列有界和代价分解。
4. 通过真实音频流实验验证：FMR、coverage、ACC/ARI/NMI、前台 P95、刷新比例、队列深度、背压次数和内存占用之间的折中关系。

不要写成“证明了增量 HDBSCAN 在所有声纹条件下保持准确率”，而应写成：

> We provide a formal, conditional consistency analysis for risk-aware selective reclustering, and empirically characterize when its geometric and local-fidelity assumptions hold on streaming speaker embeddings.

## 12. Tau Ceti 的验证范围和缺口

当前 Tau Ceti 已形式化验证：

- 严格的最近/次近距离关系推出最近候选唯一；
- 度量空间三角不等式推出距离上界、距离下界和 margin 下界；
- `2(eta+rho)<=Delta` 推出非负 margin；
- 饱和入队和截断队列更新保持容量上界；
- FAST/REFRESH 两分支满足参考契约时，单步和多步输出保持参考一致。

仍必须由实验或额外建模支持：

- 声纹嵌入是否满足给定的 `eta`、`rho`、`Delta`；
- 代表点近邻是否足以复现 HDBSCAN/FISHDBC 的密度连通决策；
- 固定阈值是否在不同说话人、噪声、漂移和新说话人条件下满足 FMR 约束；
- 参考聚类器自身的错误是否会被本文继承；
- 训练/测试流上的 ACC、ARI、NMI 和 P95 是否具有跨数据集泛化性。

因此完整、专业且可审查的结论是：

```text
形式化证明：控制层和几何不等式成立；
理论假设：已显式列出，未把假设冒充定理；
算法保证：在假设成立且提交前刷新完成时，与固定参考聚类器一致；
经验效果：必须由统一数据、统一归一化、统一指标和压力实验验证。
```

对应形式化文件：

`/home/user/Desktop/paper/TauCeti/TauCeti/Streaming/RiskAware.lean`
