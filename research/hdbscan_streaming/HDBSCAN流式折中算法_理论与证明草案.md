# 几何证书驱动的风险约束选择性重聚类：理论框架与形式化证明

## 1. 论文要回答的问题

本文研究一个流式说话人识别中的选择性计算问题：

> 在持续到达的说话人嵌入流上，如何让稳定样本以低延迟完成分配，同时把昂贵的密度聚类维护集中给真正不确定的样本，并优先抑制跨说话人误合并？

因此，`Adaptive-FISHDBC-Async` 可以被统一刻画为一个**几何证书驱动的风险约束选择性重聚类框架**：

```text
嵌入到达
  -> 提取距离、margin、簇内离散度、老化和新簇证据
  -> 稳定样本走 FAST 快路径
  -> 不确定样本走 REFRESH 刷新路径
  -> 后台 FISHDBC 维护，提交前等待相关状态完成
```

本文的核心理论命题是：

> 对于具有可观测几何证书的稳定样本，系统可以将昂贵的密度重聚类替换为低成本快路径；对于无法获得足够证书的样本，系统通过刷新路径恢复参考聚类语义。在给定风险预算的策略类内，接受全部可认证快路径是计算代价最优的门控策略。

这一定义将方法从“缓存加速技巧”提升为一个可分析的**风险约束在线决策问题**。全文理论链为：

```text
问题定义
  -> 流式状态和分区语义
  -> 几何假设
  -> 距离与 margin 引理
  -> 快路径可接受性推论
  -> 局部保真性与风险证书
  -> 逐前缀一致性主定理
  -> 风险约束下的门控最优性
  -> 队列有界与端到端代价分解
  -> 外层系统参数优化
```

这条链中，几何分析刻画可认证的稳定区域；风险证书把几何状态映射为可控制的局部失配风险；门控定理给出风险约束下的最大快路径覆盖；一致性定理连接快路径、刷新路径与参考聚类语义；资源命题将理论收益映射为前台延迟、刷新比例和队列代价。由此，方法的速度收益和质量控制来自同一个理论对象，而不是来自彼此独立的工程启发式。

## 1.1 理论贡献概览

本文理论部分可概括为四个相互衔接的结果：

1. **几何认证定理。** 样本漂移、代表点误差和说话人间隔共同决定最近/次近代表点的 margin 下界，从而给出快路径的可认证区域。
2. **风险传递定理。** 在局部保真性和刷新最终性成立时，提交分区保持与参考聚类器的分区等价；在概率保真性下，快路径失配风险被几何证书显式上界。
3. **门控最优性定理。** 在给定风险预算的可行策略类内，所有可认证状态都走 FAST、其余状态走 REFRESH 的门控策略最大化快路径覆盖率，并最小化理想化平均计算代价。
4. **系统代价定理。** 队列容量、前台延迟和后台维护成本可以被分解和约束，形成从局部门控到端到端系统参数选择的双层优化目标。

因此，本文不是仅仅展示一个更快的实现，而是给出一套从几何状态、风险证书到系统决策的统一分析框架。

## 2. 符号、参考对象和输出语义

设归一化说话人嵌入流为：

```math
X_T=(x_1,x_2,\ldots,x_T), \qquad x_t\in\mathbb{R}^d,
\qquad \|x_t\|_2=1.
```

令 `P_t^R` 表示参考聚类器在前缀 `X_t` 上产生的**样本分区**，而不是带有任意整数编号的簇标签：

```math
P_t^R=R(X_t).
```

对任意分区 `P`，定义：

```math
i\sim_P j \iff i,j \text{ belong to the same block of }P.
```

两个分区 `P` 和 `Q` 称为等价，记为 `P\cong Q`，当且仅当：

```math
\forall i,j,\quad i\sim_P j\iff i\sim_Q j.
```

这个定义消除了 HDBSCAN 簇编号的置换歧义。`R` 必须在论文中固定。若主实验把 FISHDBC 作为在线参考，就在理论中统一写作 `R=FISHDBC`；Full HDBSCAN 可以作为外部质量基准，但不能在同一个定理里交替充当参考对象。

本文讨论的是**提交时状态**。设 `\widehat P_t` 是系统在时间 `t` 允许提交的分区，而不是后台尚未收敛时产生的临时标签。系统目标是：

```math
\widehat P_t\cong P_t^R.
```

若实验不要求和参考分区逐样本一致，则理论目标退化为受约束的误合并风险：

```math
\operatorname{FMR}(\widehat P_t)\le\varepsilon.
```

这个区分很重要：后台任务完成前的 tentative label 只能用于衡量前台决策延迟，不能直接作为最终聚类质量或身份安全性的证据。

对活动簇 `k`，系统维护压缩状态：

```math
r_{k,t}=normalize\left(\sum_{i\in C_{k,t}}x_i\right),
```

```math
s_{k,t}=\frac{1}{|C_{k,t}|}\sum_{i\in C_{k,t}}\|x_i-r_{k,t}\|_2.
```

其中 `r_{k,t}` 是快路径代表点，`s_{k,t}` 是簇内离散度。它们是用于决策的缓存状态，不等价于 HDBSCAN 的完整互可达距离图或密度树。因而，后文把“代表点几何稳定”与“参考聚类分区保持不变”分成两个不同层次的命题。

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

令 `S_{t-1}` 表示到达 `x_t` 之前的压缩聚类状态。两条路径分别是：

```math
S_t^F=F(S_{t-1},x_t),\qquad
S_t^R=G(S_{t-1},x_t).
```

其中 `F` 是低成本代表点更新，`G` 是 FISHDBC 刷新或等价的风险处理路径。门控后的状态为：

```math
S_t=
\begin{cases}
S_t^F, & g_t=1,\\
S_t^R, & g_t=0.
\end{cases}
```

系统提交的是 `S_t` 所诱导的分区 `\widehat P_t`。在异步实现中，`g_t=1` 的样本可以先产生 tentative label，但只有当所有影响该分区的后台任务完成后，才允许将 `S_t` 作为 committed state。

连续风险分数可用于校准：

```math
\mathcal{R}_t=
w_d[d_{1,t}/\delta]_+
+w_m[(\mu-m_t)/\mu]_+
+w_s[s_{k_t,t}/\sigma_{max}]_+
+w_a a_t/H
+w_n n_t.
```

论文主逻辑建议使用硬门控，连续分数用于阈值扫描、可视化和消融实验。

## 4. 假设体系

为了避免把实验现象伪装成定理，假设分成四类：几何稳定性、局部保真性、风险完备性和后台服务能力。

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

`\eta` 表示局部嵌入漂移，`\rho` 表示代表点误差，`\Delta` 表示跨说话人中心间隔。

### 假设 L0：确定性局部参考等价

在最强的一致性结论中，假设存在一个可验证的局部区域 `\mathcal{A}_0`，当状态满足：

```math
z_t\in\mathcal{A}_0
\quad\Longrightarrow\quad
\operatorname{part}\big(F(S_{t-1},x_t)\big)
\cong
\operatorname{part}\big(R(X_t)\big).
```

这里的 `\operatorname{part}` 表示从聚类状态提取样本分区。该假设是从“代表点几何稳定”到“参考聚类分区不变”的桥接条件。几何分离本身不能推出 HDBSCAN 的密度连通性等价，因此它不能被写成几何定理，而必须通过校准实验或额外模型分析支持。

### 假设 Lε：概率局部保真性

更一般地，允许快路径存在小概率的不一致。令 `E_t` 表示快路径分区与参考分区不等价：

```math
E_t=
I\!\left[
\operatorname{part}\big(F(S_{t-1},x_t)\big)
\not\cong
\operatorname{part}\big(R(X_t)\big)
\right].
```

假设存在可由校准集估计的函数 `b(z)`，使得在几何证书有效的区域内：

```math
\Pr(E_t=1\mid z_t=z,\ q_t\ge0)\le b(z).
```

`b(z)` 必须是保守的风险上界，而不是仅在训练流上计算出来的经验错误率。`L0` 是 `b(z)=0` 的确定性特例，后文的概率风险定理使用 `Lε`。

### 假设 C：风险完备和刷新最终性

风险完备性要求所有可能导致跨说话人误合并的状态都满足以下至少一项：

```math
g_t=0
```

或在最多 `H` 个样本后强制刷新。

刷新最终性要求：最终提交 `\widehat P_t` 之前，所有会影响该分区的刷新任务已经完成，并且：

```math
\operatorname{part}\big(G(S_{t-1},x_t)\big)
\cong
\operatorname{part}\big(R(X_t)\big).
```

### 假设 S：后台服务能力

令刷新任务平均到达率为 `\lambda_{arrival}`，后台服务率为 `\mu_{worker}`。长期低积压需要：

```math
\lambda_{arrival}<\mu_{worker}.
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

**理论含义。** 推论 1 识别出一个由几何状态定义的 **geometrically admissible fast path**。该结果不是对 HDBSCAN 密度树的重新证明，而是为选择性计算提供了明确的认证接口：几何证书负责判断何时可以进入快路径，局部保真性负责把该证书连接到参考分区语义。中文可称为“几何认证快路径”。

## 6. 主定理：逐前缀参考一致性

### 定义 1：提交前缀

称前缀 `X_t` 已提交，当且仅当系统已经等待完所有影响其分区状态的刷新任务。只有提交前缀上的 `\widehat P_t` 才进入最终指标。

### 定理 1：条件式逐前缀一致性

若假设 L0、C 成立，且所有输出都只在提交前缀上产生，则对任意 `t`：

```math
\widehat P_t\cong P_t^R.
```

**证明。** 对 `t` 做归纳。

**基例。** `t=1` 时，若 `g_1=1`，由假设 L0 有
`\operatorname{part}(F(S_0,x_1))\cong P_1^R`；若 `g_1=0`，由刷新最终性有
`\operatorname{part}(G(S_0,x_1))\cong P_1^R`。因此
`\widehat P_1\cong P_1^R`。

**归纳步。** 假设前 `t-1` 个已经提交的前缀满足结论。处理 `X_t`：

- 若 `g_t=1`，由局部参考等价，`\operatorname{part}(S_t^F)\cong P_t^R`；
- 若 `g_t=0`，系统等待相关刷新完成，由刷新最终性，`\operatorname{part}(S_t^R)\cong P_t^R`。

两种情况都成立，因此结论对 `t` 成立。由数学归纳法，所有提交前缀均满足一致性。

### 推论 2：误合并风险的传递

若参考聚类器 `R` 在提交前缀上不存在跨说话人误合并，则本文系统也不存在跨说话人误合并。

**证明。** 由定理 1，本文输出分区和 `R` 输出分区逐前缀等价；将参考输出的无误合并性质代入即可。

该推论的理论价值在于把系统风险归因清晰化：方法不会额外引入超出参考聚类器的误合并风险；参考聚类器自身的质量则由外部基准和实验协议独立刻画。

## 7. 从几何稳定性到可验证风险上界

前面的几何推导只给出了“最近代表点不会轻易被竞争簇超过”的充分条件。要把它变成算法决策依据，还需要把几何条件、簇状态和历史老化统一写成一个可校准的风险证书。定义几何证书裕量：

```math
q_t=\Delta_t-2(\eta_t+\rho_t),
```

其中 `\Delta_t` 是当前样本对应簇与最危险竞争簇之间的估计中心间隔，`\eta_t` 是局部漂移上界，`\rho_t` 是代表点误差上界。令：

```math
z_t=(d_{1,t},m_t,s_{k_t,t},a_t,n_t,q_t)
```

是门控可观测状态。由校准集估计一个风险证书 `b(z)`，满足置信度至少为 `1-\alpha` 的条件风险上界：

```math
\Pr(E_t=1\mid z_t=z,\ q_t\ge0)\le b(z),
```

其中 `E_t=1` 表示快路径分区与参考分区不等价。这里的 `b(z)` 必须是保守的上置信界，而不是仅在训练流上计算得到的经验错误率。可采用结构化的上界：

```math
b(z)=\min\!\left\{1,\;
b_{geom}(q)+b_{drift}(s,a)+b_{novel}(n)\right\},
```

并令各项在几何裕量增大、簇内离散度减小、状态更新较新以及新簇证据消失时不增。于是，快路径的“可接受”不再只是一个经验阈值，而是一个可审计条件：

```math
\mathcal{A}_\varepsilon
=\{z:q(z)\ge0,\ b(z)\le\varepsilon\}.
```

一种可复现的校准方式是将独立校准流按 `z` 的区间划分为若干 bin `B_\ell`。在第 `\ell` 个 bin 中观察到 `n_\ell` 个样本和 `e_\ell` 个快路径不一致事件，并用二项分布的一侧上置信界 `\overline b_\ell(\alpha_\ell)` 估计该 bin 的风险：

```math
\widehat b(z)=\overline b_\ell(\alpha_\ell),
\qquad z\in B_\ell,
\qquad
\sum_\ell\alpha_\ell\le\alpha.
```

通过并合界，在置信度至少为 `1-\alpha` 下，所有校准 bin 的真实条件风险同时不超过对应上置信界。这样，`\mathcal{A}_\varepsilon` 可以由独立校准数据确定，而不是由测试集标签反向选择。

### 定理 2：快路径的条件风险保证

若假设 Lε 的风险证书在校准置信度 `1-\alpha` 下有效，则对所有满足 `z_t\in\mathcal{A}_\varepsilon` 的快路径样本：

```math
\Pr(E_t=1\mid z_t)\le\varepsilon,
\qquad z_t\in\mathcal{A}_\varepsilon.
```

**证明。** 对任意 `z_t\in\mathcal{A}_\varepsilon`，由集合定义有 `q_t\ge0` 且 `b(z_t)\le\varepsilon`。由风险证书有效性：
`\Pr(E_t=1\mid z_t)\le b(z_t)\le\varepsilon`。对被门控到快路径的状态分布取条件期望，得到结论。若采用假设 L0，则 `E_t=0` 几乎处处成立，定理退化为定理 1 所需的确定性局部等价。

这个定理建立了从几何状态到系统决策的统一接口：几何分析产生 `q_t`，校准过程产生 `b(z)`，门控策略在风险预算内选择快路径。实验因此可以直接报告证书覆盖率、校准可靠性和真实 FMR，形成对理论机制的闭环验证。

## 8. Tradeoff 最优性：在风险约束下尽可能多走快路径

### 定义 2：可证明策略类

考虑只在 `FAST` 和 `REFRESH` 之间选择的策略类 `\Pi_\varepsilon`。策略 `\pi\in\Pi_\varepsilon` 必须满足：

```math
\pi(z)=FAST\Longrightarrow z\in\mathcal{A}_\varepsilon.
```

因此，当状态不属于 `\mathcal{A}_\varepsilon` 时，所有可证明安全的策略都只能选择 `REFRESH`。这一定义把正确性约束放在策略类中，而不是事后用平均 accuracy 替代风险约束。

设单样本前台代价满足：

```math
c_f<c_r,
```

其中 `c_f` 是代表点快路径代价，`c_r` 是等待或触发刷新路径的等效代价。更一般地，也可以令二者依赖于状态 `z`，只要求在可接受区域内 `c_f(z)\le c_r(z)`。为便于给出清晰的闭式结论，下面先采用常数代价模型。定义策略的快路径覆盖率：

```math
\kappa(\pi)=\Pr(\pi(z)=FAST),
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

因此，`\pi^*` 是该风险约束策略类中的最大覆盖门控，也是平均计算代价最小的门控。换言之，给定风险预算后，任何额外拒绝可认证状态的策略都会牺牲可获得的快路径覆盖，而任何接纳不可认证状态的策略都会离开可行域。该结论给出了本文 tradeoff 设计的理论基准。

**证明。** 对任意状态 `z`：若 `z\notin\mathcal{A}_\varepsilon`，可行性要求所有 `\pi\in\Pi_\varepsilon` 都选择 `REFRESH`，与 `\pi^*` 相同；若 `z\in\mathcal{A}_\varepsilon`，`\pi^*` 选择 `FAST`，其他可行策略至多选择 `FAST`，因此逐状态有：

```math
I[\pi(z)=FAST]\le I[\pi^*(z)=FAST].
```

对状态分布取期望得到 `\kappa(\pi)\le\kappa(\pi^*)`。代入 `J(\pi)=c_r-(c_r-c_f)\kappa(\pi)`，并利用 `c_r-c_f>0`，得到 `J(\pi^*)\le J(\pi)`。若代价依赖于状态，只需在每个可接受状态上比较 `c_f(z)\le c_r(z)`，同样的逐状态交换证明仍然成立。

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

这给出了论文所需的 tradeoff 方向：放宽风险预算单调扩大可认证快路径区域，因而提高快路径覆盖率并降低理想化平均计算代价；收紧风险预算则以更多刷新或更低覆盖率为代价。系统实验进一步将该理论 Pareto 前沿投影到 `FMR`、`coverage`、`P95` 和 `refresh ratio`。

### 推论 4：固定覆盖目标下的最小风险选择

如果实验还要求 `\kappa(\pi)\ge\gamma`，则应在满足覆盖率的策略中选择最小风险的状态集合，而不是简单提高单个距离阈值。一个可操作的版本是令状态按 `b(z)` 从小到大排序，优先将风险证书最强的状态交给 `FAST`，直到达到 `\gamma`。这相当于求解：

```math
\min_{A}\quad \mathbb{E}[b(z_t)\mid z_t\in A]
\qquad
\text{s.t.}\quad \Pr(z_t\in A)\ge\gamma.
```

**证明。** 设可行集合 `A` 中包含状态 `z_h`，但集合外存在 `z_l`，且 `b(z_l)<b(z_h)`。将 `z_h` 替换为 `z_l` 不改变集合的测度，因此仍满足 `P(z_t\in A)\ge\gamma`；同时集合内风险总和严格下降，集合测度不变，所以条件平均风险严格下降。反复执行该交换，直到集合包含所有更低风险状态而不包含更高风险状态，所得集合必为风险阈值集合 `A_\tau=\{z:b(z)\le\tau\}`，必要时在边界处随机取样以精确满足覆盖率。故风险证书排序是固定覆盖目标下的最小风险选择。这一结果解释了为什么连续风险证书应参与阈值扫描，即使线上实现仍采用硬门控。

需要区分两个结论：定理 3 证明的是固定风险预算下的最低代价，推论 4 证明的是固定快路径覆盖率下的最低证书风险。二者合起来才构成本文的 tradeoff 理论，而不是单独声称“Adaptive 一定优于 Full”。

## 9. 资源和延迟命题

令后台队列容量为 `B`，任务到达数和完成数分别为 `u_t,v_t`，令 `Q_t` 表示第 `t` 个时刻的队列长度：

```math
Q_{t+1}=\min\{B,\;\max\{0,\;Q_t+u_t-v_t\}\}.
```

### 命题 1：队列有界

若 `Q_0\in[0,B]`，则对任意 `t`：

```math
0\le Q_t\le B.
```

**证明。** `\max\{0,\cdot\}` 保证下界，`\min\{B,\cdot\}` 保证上界；对每个时间步递推即可。Tau Ceti 对该截断更新以及单任务入队特例进行形式化检查。

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

该式揭示方法收益来源：只有当 `p_r` 足够小，且后台维护不阻塞前台，选择性计算才会优于每样本完整刷新。对于规模随时间增长的流，论文应直接报告 `c_m(n_t)`、刷新次数和实际前台 P95，而不能把该近似式当作渐近复杂度定理。

## 10. 统一优化目标

令：

```math
\theta=(\delta,\mu,\sigma_{max},H,B,B_{batch}),
```

其中 `B_{batch}` 是后台批大小，避免与风险证书 `b(z)` 混淆。定义跨说话人误合并率：

```math
FMR=
\frac{\sum_{i<j}I[y_i\ne y_j]I[i\sim_{\widehat P_T}j]}
{\sum_{i<j}I[y_i\ne y_j]}.
```

这里 `\widehat P_T` 只包含被系统合并到非噪声簇的样本；被拒识或标记为 noise 的样本不会被计入误合并事件。

令 `L_t^{front}(\theta)` 为前台延迟，`C_t^{maint}(\theta)` 为后台维护成本，`M_t(\theta)` 为内存占用，`I_t^{refresh}(\theta)` 为是否触发刷新。则一条长度为 `T` 的流的运行代价可写为：

```math
C_T(\theta)=
\frac{1}{T}\sum_{t=1}^{T}
\left[
L_t^{front}(\theta)
+\lambda_m M_t(\theta)
+\lambda_c C_t^{maint}(\theta)
+\lambda_r I_t^{refresh}(\theta)
\right].
```

其中各项应先按部署预算或基准方法归一化，否则不同量纲的加权和没有可解释性。

系统参数选择是一个外层约束优化问题：

```math
\theta^*\in
\arg\min_{\theta}\mathbb{E}[C_T(\theta)].
```

约束为：

```math
FMR(\theta)\le\epsilon,
\qquad
coverage(\theta)\ge\gamma,
\qquad
\Pr\!\left(\max_{t\le T}Q_t(\theta)\le B\right)\ge1-\alpha_Q.
```

这个目标应理解为两层优化。内层固定风险预算 `\varepsilon`，在 `\Pi_\varepsilon` 中选择最大可行门控 `\pi_\varepsilon^*`；外层再通过数据集和压力实验选择 `\theta`，使风险、覆盖率、队列和端到端延迟同时满足约束。定理 3 只证明内层门控的条件最优性，不宣称外层参数或整个 Adaptive-FISHDBC-Async 对所有数据都全局最优。

这正面呼应论文核心目标：在误合并风险受控、有效覆盖率不低于目标值且后台队列有界的前提下，减少前台延迟和重复密度聚类计算。

## 11. 论文中的理论贡献表述

建议将理论贡献包装为 **Geometry-Certified Risk-Constrained Selective Reclustering**，正文可写成：

1. We formulate streaming speaker clustering as a risk-constrained selective computation problem, where each embedding is routed to either a low-cost representative update or a density-aware refresh path.
2. We derive a geometry certificate that combines local embedding drift, representative error, and inter-speaker separation to characterize the admissible fast-path region.
3. We establish a conditional partition-consistency result: when the local fidelity and refresh contracts hold, committed outputs remain equivalent to a fixed reference clusterer on every prefix.
4. We prove that, within the admissible FAST/REFRESH policy class, accepting every certified state is optimal for fast-path coverage and idealized computation cost under a prescribed risk budget.
5. We formulate system parameter selection as an outer constrained optimization over false merges, coverage, queue capacity, latency, and maintenance cost, and validate the resulting risk-cost frontier on streaming speaker embeddings.

理论主张的推荐总述是：

> We develop a geometry-certified, risk-constrained theory for selective reclustering on streaming speaker embeddings. The theory identifies when representative-based fast updates preserve the reference partition, proves the optimality of admitting all certified states within the FAST/REFRESH policy class, and exposes the resulting latency-risk tradeoff through an outer constrained optimization.

中文概括可写为：

> 本文建立一种几何证书驱动的风险约束选择性重聚类理论，将流式说话人聚类中的“是否重聚类”转化为可认证的在线门控问题。该理论一方面通过漂移、代表点误差和簇间隔刻画快路径的可行区域，另一方面证明在给定风险预算的 FAST/REFRESH 策略类内，接纳全部可认证状态具有最大快路径覆盖率和最小理想化计算代价。

### 证明依赖关系

正文中的证明应按以下依赖关系组织，避免把引理和定理写成彼此孤立的公式：

```text
假设 G
  -> 引理 1：同簇代表点距离上界
  -> 引理 2：异簇代表点距离下界
  -> 引理 3：最近/次近代表点 margin 下界
  -> 推论 1：几何可接受快路径
  -> 假设 L0
  -> 定理 1：提交分区与参考分区逐前缀等价
  -> 推论 2：参考聚类器的误合并性质可被传递

假设 G + 独立校准集
  -> 风险证书 b(z)
  -> 定理 2：快路径条件风险不超过 epsilon
  -> 定义 Pi_epsilon
  -> 定理 3：风险约束下最大快路径覆盖与最小理想化代价
  -> 推论 3/4：风险预算 Pareto 方向与固定 coverage 下的最小证书风险

定理 3 + 队列模型
  -> 命题 1：队列容量有界
  -> 命题 2/3：前台延迟和后台计算成本分解
  -> 外层参数约束优化
```

因此，论文理论部分的核心不是声称“快路径天然等价于 HDBSCAN”，而是建立如下可审查的条件链：几何条件产生候选快路径，校准程序把候选状态转化为风险证书，风险门控在证书约束下获得策略类内最优性，后台刷新和提交协议保证最终输出语义，实验再检验这些条件在语音流上是否成立。

## 12. Tau Ceti 的验证范围和缺口

当前 Tau Ceti 已形式化验证以下理论骨架：

- 严格的最近/次近距离关系推出最近候选唯一；
- 度量空间三角不等式推出距离上界、距离下界和 margin 下界；
- `2(\eta+\rho)\le\Delta` 推出非负 margin；
- 饱和入队和截断队列更新保持容量上界；
- FAST/REFRESH 两分支满足参考契约时，单步和多步分区输出保持参考一致；
- 在“认证状态允许 FAST、非认证状态必须 REFRESH、且 FAST 成本更低”的离散策略类中，最大可行门控具有点态成本最优性。

因此，后续实验不是重新寻找一个模糊的“是否有效”，而是要完成理论参数的实例化和校准：

- 声纹嵌入是否满足给定的 `\eta`、`\rho`、`\Delta`；
- 代表点近邻是否足以复现 HDBSCAN/FISHDBC 的密度连通决策；
- 风险证书 `b(z)` 是否能在独立校准流上提供有效上置信界；
- 固定阈值是否在不同说话人、噪声、漂移和新说话人条件下满足 FMR 约束；
- 参考聚类器自身的错误是否会被本文继承；
- 训练/测试流上的 ACC、ARI、NMI 和 P95 是否具有跨数据集泛化性。

论文中建议将最终理论结论表述为：

```text
The proposed method is a geometry-certified selective reclustering policy.
Its fast path is admitted by an explicit risk certificate rather than a heuristic
similarity threshold. Under the local-fidelity and refresh contracts, committed
prefixes preserve the reference partition. Among policies that satisfy the same
risk budget, the maximal admissible gate maximizes fast-path coverage and
minimizes idealized computation cost. The outer operating point is selected by
the empirical FMR-coverage-latency-memory frontier.
```

对应形式化文件：

`/home/user/Desktop/paper/TauCeti/TauCeti/Streaming/RiskAware.lean`
