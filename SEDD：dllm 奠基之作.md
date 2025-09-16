# 一：Tweedie τ-leaping推导讲解（来自DS）

本文涉及两个定理：**Theorem 4.1**（离散 Tweedie 定理）和 **Theorem 4.2**（Tweedie τ-leaping 的最优性）。我将逐步推导这些结论，并解释其背后的直觉。

## 背景

- 扩散过程由 ODE $dp_t = Q p_t$ 描述，其中 $Q$ 是转移速率矩阵（例如，连续时间马尔可夫链的生成器）。
- $p_t$ 是时间 $t$ 的概率分布，$\exp(tQ)$ 是矩阵指数，表示从时间 0 到时间 $t$ 的转移矩阵。
- 真实去噪器 $p_{0|t}(x_0|x_t)$ 是从噪声观测 $x_t$ 恢复初始状态 $x_0$ 的条件概率。

## Theorem 4.1 的推导

Theorem 4.1 给出了真实去噪器的表达式：

$p_{0|t}(x_0|x_t) = \left( \exp(-tQ) \left[ \frac{p_t(i)}{p_t(x_t)} \right]^N_{i=1} \right)_{x_0} \exp(tQ)(x_t, x_0)$

**推导步骤**：

1. **贝叶斯定理**：真实后验概率为：
    
    $p_{0|t}(x_0|x_t) = \frac{p_{t|0}(x_t|x_0) p_0(x_0)}{p_t(x_t)}$
    
    其中 $p_{t|0}(x_t|x_0) = \exp(tQ)(x_0, x_t)$ 是从 $x_0$ 到 $x_t$ 的前向转移概率。
    
2. **前向与反向关系**：由扩散 ODE $p_t = \exp(tQ) p_0$，可得 $p_0 = \exp(-tQ) p_t$。因此，对于初始状态 $x_0$：
    
    $p_0(x_0) = \sum_i \exp(-tQ)(x_0, i) p_t(i)$
    
    代入贝叶斯公式：
    
    $p_{0|t}(x_0|x_t) = \frac{ \exp(tQ)(x_0, x_t) \sum_i \exp(-tQ)(x_0, i) p_t(i) }{ p_t(x_t) }$
    
3. **对称性假设**：如果转移矩阵 $Q$ 是对称的（即 $\exp(tQ)$ 对称），则 $\exp(tQ)(x_0, x_t) = \exp(tQ)(x_t, x_0)$。这意味着：
    
    $p_{0|t}(x_0|x_t) = \left( \sum_i \exp(-tQ)(x_0, i) \frac{p_t(i)}{p_t(x_t)} \right) \exp(tQ)(x_t, x_0)$
    
    这正是定理中的表达式，其中 $\left[ \frac{p_t(i)}{p_t(x_t)} \right]^N_{i=1}$ 是向量，$\exp(-tQ)$ 作用其上后取 $x_0$ 分量。
    

**结论**：Theorem 4.1 在 $Q$ 对称的条件下成立，提供了从噪声中恢复原始数据的理论工具。

## Theorem 4.2 的推导

Theorem 4.2 涉及 τ-leaping 策略，其中 token 更新是独立且同时进行的。公式 (19) 定义了更新规则：

$\left( \exp(-\sigma^{\Delta t}_t Q)s_\theta(\mathbf{x}_t, t)_i \right)_{x^i_{t-\Delta t}} \exp(\sigma^{\Delta t}_t Q)(x^i_t, x^i_{t-\Delta t})$

其中 $\sigma^{\Delta t}_t = \overline{\sigma}(t) - \overline{\sigma}(t - \Delta t)$ 是时间间隔，$s_\theta$ 是评分函数。

**推导步骤**：

1. **真实反向转移**：对于单个 token，从时间 $t$ 到 $t-\Delta t$ 的真实反向转移概率为：
    
    $p_{t-\Delta t|t}(x_{t-\Delta t}|x_t) = \frac{ p_{t|t-\Delta t}(x_t|x_{t-\Delta t}) p_{t-\Delta t}(x_{t-\Delta t}) }{ p_t(x_t) }$
    
    其中 $p_{t|t-\Delta t}(x_t|x_{t-\Delta t}) = \exp(\sigma^{\Delta t}_t Q)(x_{t-\Delta t}, x_t)$。使用 $p_{t-\Delta t} = \exp(-\sigma^{\Delta t}_t Q) p_t$，可得：
    
    $p_{t-\Delta t|t}(x_{t-\Delta t}|x_t) = \left( \sum_j \exp(-\sigma^{\Delta t}_t Q)(x_{t-\Delta t}, j) \frac{p_t(j)}{p_t(x_t)} \right) \exp(\sigma^{\Delta t}_t Q)(x_{t-\Delta t}, x_t)$
    
    这与 Theorem 4.1 形式类似，但针对时间间隔 $\Delta t$.
    
2. **评分函数的作用**：如果评分函数 $s_\theta$ 是完美的，则对于每个 token $i$ 和状态 $j$，有：
    
    $s_\theta(\mathbf{x}_t, t)_i (j) = \frac{p_t(\mathbf{x}_t^{-i}, x_t^i = j)}{p_t(\mathbf{x}_t)}$
    
    其中 $\mathbf{x}_t^{-i}$ 是除第 $i$ 个 token 外的序列。如果扩散过程在 token 之间独立（即 $p_t(\mathbf{x}_t) = \prod_i p_t^i(x_t^i)$），则该比值退化为边际比值 $\frac{p_t^i(j)}{p_t^i(x_t^i)}$.
    
3. **τ-leaping 中的独立更新**：在 τ-leaping 中，我们假设 token 更新独立，因此联合更新概率为：
    
    $p_t^{tweedie}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) = \prod_i q_i(x_{t-\Delta t}^i | x_t^i)$
    
    其中 $q_i$ 由公式 (19) 定义。如果 $s_\theta$ 完美，则公式 (19) 给出每个 token 的真实反向转移概率，因此 $p_t^{tweedie}$ 匹配真实反向分布 $p_{t-\Delta t|t}$.
    
4. **KL 散度最小化**：由于真实反向分布可能因子化（当 token 独立时），独立更新策略的最优近似是使每个 token 的边际分布匹配真实边际分布。因此，公式 (19) 在所有 τ-leaping 策略中最小化 KL 散度。
    

**结论**：Theorem 4.2 在评分函数完美且扩散过程 token 独立的条件下成立，确保了 τ-leaping 策略的最优性。

## 统一算法（Algorithm 2）

文本提到这些模拟算法被统一在 Algorithm 2 中。虽然未给出详细步骤，但 Algorithm 2 likely 是一个基于 τ-leaping 和 Tweedie 去噪的扩散模型采样算法，使用公式 (19) 进行高效且最优的采样。

## 总结

- Theorem 4.1 提供了离散扩散中真实去噪器的解析形式。
- Theorem 4.2 shows that with a perfect score function, the τ-leaping update rule minimizes KL divergence with the true reverse process.
- 这些结论依赖于扩散过程的对称性和 token 独立性假设，适用于文本扩散模型等场景。

如果您需要更详细的数学推导或有特定问题，请随时告知！

# 二：采样示例（来自Grok3）
下面我们将为 **SEDD Absorb** 版本（即使用吸收状态转移矩阵 $Q^{\text{absorb}}$，也称为mask版本）重新举例，详细且严谨地计算使用 **Tweedie $\tau$-leaping** 采样的过程。SEDD Absorb 通过引入 MASK 吸收状态（类似于 BERT 的掩码机制）来模拟扩散过程，其中非 MASK 状态会逐步转移到 MASK 状态作为基础分布 $p\_T \approx p\_{\text{base}}$（全 MASK）。以下计算基于论文页面4的公式16（$Q^{\text{absorb}}$ 定义）和页面5的公式19（Tweedie $\tau$-leaping），以及整体反向过程框架（公式8）。

### 步骤 1: 问题设置

* **序列结构**: 假设简单序列 $x = (x\_1, x\_2)$，每个 token $x\_i$ 从集合 ${0, 1, \text{MASK}}$ 中取值（$N = 3$，MASK 是吸收状态）。总状态空间为 $3^2 = 9$ 种可能序列。
* **扩散矩阵 $Q^{\text{tok}}$**: 基于论文公式16，为 token 级吸收矩阵（列和为 0，非对角非负）。对于 ${0, 1, \text{MASK}}$（行/列顺序：0, 1, MASK），矩阵定义为：

$$
Q^{\text{tok}} = \begin{bmatrix}
-1 & 0 & 0 \\
0 & -1 & 0 \\
1 & 1 & 0
\end{bmatrix}.
$$

解释：从状态 0 或 1 以率 1 转移到 MASK（最后一列前两行为 0，对角 -1）；MASK 是吸收状态（最后一列全 0）。注意：论文中为一般 N，但这里简化到 $N=3$。$Q^{\text{tok}}\_t = \sigma(t) Q^{\text{tok}}$，其中 $\sigma(t)$ 为噪声水平。

* **噪声スケジュール**: 假设线性噪声率 $\sigma(t) = t$（简化计算），累计噪声为

$$
\sigma(t) = \int_0^t s \, ds = \frac{t^2}{2}.
$$

总时间 $T = 1$，步长 $\Delta t = 0.1$，共 10 步（论文中建议几何或对数线性スケジュール，但这里用线性示例）。

* **基础分布 $p\_T$**: 为全 MASK 分布（吸收极限），即 $p\_T((\text{MASK}, \text{MASK})) = 1$，其他 0（或带少量泄漏以避免无穷 KL，但这里纯吸收）。
* **初始序列**: 从 $p\_T$ 采样得到 $x\_T = (\text{MASK}, 0)$（假设部分已知或提示；纯无条件从全 MASK 开始）。
* **神经网络 $s\_\theta$**: 训练好的模型输出

$$
s_\theta(x_t, t)_{i,y} \approx \frac{p_t(x_1 \dots y \dots x_d)}{p_t(x_1 \dots x_i \dots x_d)}
$$

（仅汉明距离 1）。为示例，假设具体值（基于假设 $p\_t$ 分布）。

反向过程由 ODE 描述（公式8）：

$$
\frac{d p^\theta_{T-t}}{dt} = Q^\theta_{T-t} p^\theta_{T-t}, \quad p^\theta_T = p_{\text{base}},
$$

其中

$$
Q^\theta_t(y, x) = s_\theta(x, t)_y \, Q_t(x, y) \quad (x \ne y).
$$

Tweedie $\tau$-leaping 采样基于公式19：

$$
p_i(y \mid x^i_t) = \left[ \exp\left( -\sigma_{\Delta t}(t) Q^{\text{tok}} \right) s_\theta(x_t, t)_i \right]_y \cdot \exp\left( \sigma_{\Delta t}(t) Q^{\text{tok}} \right)(x^i_t, y),
$$

其中 $\sigma\_{\Delta t}(t) = \sigma(t) - \sigma(t - \Delta t) > 0$（公式20）。这优化了独立 token 更新下的 KL 散度（Theorem 4.2）。

### 步骤 2: 具体计算 (第一步 $t = 1$ 到 $t = 0.9$）

* **噪声差值**:

$$
\sigma(1) = \frac{1^2}{2} = 0.5, \quad
\sigma(0.9) = \frac{0.9^2}{2} = 0.405, \quad
\sigma_{\Delta t}(1) = 0.5 - 0.405 = 0.095.
$$

* **当前状态 $x\_t = (\text{MASK}, 0)$**: 计算每个位置 $i = 1, 2$ 的转移概率 $p\_i(y \mid x^i\_t)$。
* **估计 $s\_\theta(x\_t, t)\_i$**: 假设 $t=1$ 时，$p\_t$ 分布接近基础（高噪声），例如

$$
p_t((\text{MASK}, 0)) = 0.4,\quad
p_t((0, 0)) = 0.1,\quad
p_t((1, 0)) = 0.15,\quad
p_t((\text{MASK}, 1)) = 0.1,\quad
p_t((\text{MASK}, \text{MASK})) = 0.25.
$$

对于 $i=1$（$x^1\_t = \text{MASK}$，改变到 $y=0,1,\text{MASK}$）:

$$
s_\theta(x_t, 1)_{1,0} \approx \frac{p_t((0, 0))}{p_t((\text{MASK}, 0))} = \frac{0.1}{0.4} = 0.25,
$$

$$
s_\theta(x_t, 1)_{1,1} \approx \frac{p_t((1, 0))}{p_t((\text{MASK}, 0))} = \frac{0.15}{0.4} = 0.375,
$$

$$
s_\theta(x_t, 1)_{1,\text{MASK}} \approx \frac{p_t((\text{MASK}, 0))}{p_t((\text{MASK}, 0))} = 1
$$

（自比率为1，但实际仅 $y\ne x$ 时建模；为完整假设）。所以

$$
s_\theta(x_t, 1)_1 = \begin{bmatrix}0.25 \\[2pt] 0.375 \\[2pt] 1\end{bmatrix}
$$

（向量，对应 $y=0,1,\text{MASK}$）。
对于 $i=2$（$x^2\_t = 0$，改变到 $y=0,1,\text{MASK}$）：类似假设

$$
s_\theta(x_t, 1)_2 = \begin{bmatrix}1 \\[2pt] 0.25 \\[2pt] 0.625\end{bmatrix}
$$

（示例值）。

* **矩阵指数计算**: 计算

$$
\exp(\sigma_{\Delta t}(1) Q^{\text{tok}}) = \exp(0.095 \, Q^{\text{tok}}).
$$

使用 Taylor 展开（因小 $0.095$）：

$$
\exp(0.095 Q^{\text{tok}}) \approx I + 0.095 Q^{\text{tok}} + \frac{(0.095)^2}{2} (Q^{\text{tok}})^2 + \cdots.
$$

先计算 $(Q^{\text{tok}})^2$：

$$
(Q^{\text{tok}})^2 = \begin{bmatrix}
-1 & 0 & 0 \\
0 & -1 & 0 \\
1 & 1 & 0
\end{bmatrix}^2
= \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
-2 & -2 & 0
\end{bmatrix}
\quad(\text{近似}).
$$

结果近似矩阵：

$$
\exp(0.095 Q^{\text{tok}}) \approx \begin{bmatrix}
e^{-0.095} & 0 & 0 \\
0 & e^{-0.095} & 0 \\
0.095 & 0.095 & 1
\end{bmatrix}
\approx \begin{bmatrix}
0.909 & 0 & 0 \\
0 & 0.909 & 0 \\
0.095 & 0.095 & 1
\end{bmatrix}.
$$

（注：$e^{-0.095} \approx 0.909$；对于吸收，MASK 行不变。实际用闭形式：非吸收方向衰减，转移到 MASK 累积。）

* **Tweedie 转移概率** (以 $i=1$，$x^1\_t = \text{MASK}$ 为例)：
  先计算向量部分：

$$
v = \exp(- \sigma_{\Delta t}(1) Q^{\text{tok}}) \; s_\theta(x_t, t)_1
\approx \exp(-0.095 Q^{\text{tok}}) \begin{bmatrix}0.25 \\[2pt] 0.375 \\[2pt] 1\end{bmatrix}.
$$

$$
\exp(-0.095 Q^{\text{tok}}) \approx \begin{bmatrix}
0.909 & 0 & 0 \\
0 & 0.909 & 0 \\
-0.095 & -0.095 & 1
\end{bmatrix}
\quad(\text{负指数，反向}).
$$

因此

$$
v \approx \begin{bmatrix}
0.909 \cdot 0.25 \\
0.909 \cdot 0.375 \\
-0.095 \cdot 0.25 -0.095 \cdot 0.375 + 1 \cdot 1
\end{bmatrix}
= \begin{bmatrix} 0.227 \\[2pt] 0.341 \\[2pt] 0.941 \end{bmatrix}.
$$

然后

$$
p_1(y \mid \text{MASK}) = v_y \cdot \exp(0.095 Q^{\text{tok}})(\text{MASK}, y).
$$

$\exp(0.095 Q^{\text{tok}})(\text{MASK}, y)$ 是矩阵中从列 $\text{MASK}$ 到行 $y$ 的分量（矩阵的第3列）。对于 $x=\text{MASK}$（列3），$\exp$ 的第3列为 $\[0,0,1]^T$（MASK 自留1，其他0，因吸收率0）。所以

$$
p_1(0 \mid \text{MASK}) \propto 0.227 \cdot 0 = 0,
$$

$$
p_1(1 \mid \text{MASK}) \propto 0.341 \cdot 0 = 0,
$$

$$
p_1(\text{MASK} \mid \text{MASK}) \propto 0.941 \cdot 1 = 0.941.
$$

归一化后：$p\_1(\text{MASK} \mid \text{MASK}) \approx 1$，其他 $\approx 0$（高噪声时，MASK 倾向自留）。

* **采样**: 从 $p\_1(y \mid \text{MASK})$ 采样，几乎确定得到 $x^1\_{0.9} = \text{MASK}$。对于 $i=2$ ($x^2\_t=0$)，类似计算假设

$$
p_2(0\mid0)\approx0.8,\quad p_2(1\mid0)\approx0.1,\quad p_2(\text{MASK}\mid0)\approx0.1,
$$

采样得到 $x^2\_{0.9}=0$。于是 $x\_{0.9} = (\text{MASK}, 0)$。

### 步骤 3: 迭代过程

重复上述，从 $t=0.9$ 到 $0.8$ 等，逐步减小 $t$。随着 $t$ 减小，$s\_\theta$ 会引导从 MASK "去噪" 到数据 token（0或1）。例如，在低 $t$ 时，转移概率会允许从 MASK 跳回 0/1。

### 步骤 4: 结果与性质

经过 10 步，$x\_0$ 趋近 $p\_{\text{data}}$，例如最终 $x\_0 = (1, 0)$。在 Absorb 版本，采样支持 infilling（公式21-22，Algorithm 3）：固定已知位置（非 MASK），仅更新 MASK 位置。

此过程严格遵循论文，适用于语言任务（如从 MASK 序列生成文本）。在实践中，矩阵指数可闭形式计算（因结构化），效率高。


