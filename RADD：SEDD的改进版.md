# 一：吸收式离散扩散模型的去噪分数熵损失函数（来自GROK3）

这个图片展示了一个数学公式，它来自论文《Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data》（简称 RADD 论文）的相关部分。该公式是有限时间（从 0 到 T）的去噪分数熵（Denoising Score Entropy, DSE）损失函数 $\mathcal{L}_{DSE}^T(x_0)$ 的表达式，这是吸收式离散扩散模型（absorbing discrete diffusion model）在重新参数化（reparameterized）后的形式，用于训练模型。

## 公式的数学表示

公式为：

$$  
\mathcal{L}_{DSE}^{T}(x_0) = \int_0^T \mathbb{E}_{x_t \sim p_{t|0}(x_t | x_0)} \left[ \sum_{x_i = [M], j \neq [M]} \sigma(t) \left( \frac{e^{-\bar{\sigma}(t)}}{1 - e^{-\bar{\sigma}(t)}} c_\theta(x_t)[i,j] \right) - \frac{e^{-\bar{\sigma}(t)}}{1 - e^{-\bar{\sigma}(t)}} \mathbb{I}(x_0^i = j) \log \left( \frac{e^{-\bar{\sigma}(t)}}{1 - e^{-\bar{\sigma}(t)}} c_\theta(x_t)[i,j] \right) + K \left( \frac{e^{-\bar{\sigma}(t)}}{1 - e^{-\bar{\sigma}(t)}} \mathbb{I}(x_0^i = j) \right) \right] dt  
$$

其中：

- $x_0$ 是干净数据（clean data）。
- $x_t$ 是时间 t 处的噪声样本，从前向过程 $p_{t|0}(x_t | x_0)$ 中采样。
- $[M]$ 是掩码令牌（mask token），表示吸收状态。
- 求和针对所有掩码位置 i（即 $x_i = [M]$）和可能的非掩码令牌 j（$j \neq [M]$）。
- $\sigma(t)$ 是噪声调度函数（noise schedule），$\bar{\sigma}(t) = \int_0^t \sigma(s) , ds$ 是累积噪声。
- $c_\theta(x_t)[i,j]$ 是模型参数化网络的输出，近似干净数据的条件概率 $p_0(x^i = j | x_t^{UM})$，其中 $x_t^{UM}$ 是 $x_t$ 中所有未掩码的令牌。
- $\mathbb{I}(x_0^i = j)$ 是指示函数，如果干净数据在位置 i 是 j，则为 1，否则为 0。
- $K(a) = a \log a - a$ 是一个常量函数，用于分数匹配或 KL 散度的部分。

## 公式的含义和背景

### 上下文

这个公式出现在论文的第 3.3 节（统一吸收式离散扩散和任意顺序自回归模型）附近，或其证明附录中。它是基于 Theorem 1（吸收扩散中的解析混凝土分数）的重新参数化形式。在原始的 SEDD 模型中，混凝土分数（concrete score） $s_\theta$ 是时间相关的，但根据 Theorem 1，它可以分解为一个时间相关标量 $\frac{e^{-\bar{\sigma}(t)}}{1 - e^{-\bar{\sigma}(t)}}$ 乘以时间无关的条件概率 $c_\theta$。

### 作用

DSE 损失是扩散模型训练的目标函数的上界，用于估计负对数似然（negative log-likelihood）。它通过最小化这个积分来训练模型，使模型学习从噪声样本 $x_t$ 恢复干净数据 $x_0$。

- **第一项**：线性项，涉及模型预测 $c_\theta$。
- **第二项**：类似于交叉熵（cross-entropy）项，促进模型预测匹配真实条件分布。
- **第三项**：常量 K 项，确保损失是分数匹配的变体。

### 为什么重新参数化

原始分数网络需要处理时间依赖，但 RADD 通过移除时间条件简化模型，只学习干净数据的条件分布 $q_\theta \approx p_0$，这提高了优化效率、采样速度（通过缓存），并在零-shot 语言建模基准上取得了 SOTA 性能。

### 与论文其他部分的联系

这个公式等价于 t-DCE 损失（通过移除某些项得到），并进一步等价于 λ-DCE 和 AO-ARM 的损失（Theorem 2），实现了扩散模型与自回归模型的统一。

如果需要更详细的推导或论文特定页面的内容，可以提供更多细节！


# 二：解释一下$\frac{e^{-\bar{\sigma}(t)}}{1 - e^{-\bar{\sigma}(t)}} \mathbb{I}(x_0^i = j)$的含义
因为该部分应该为$\mathbf{\frac{p_{t|0}(\hat{x}_t \mid x_0)}{p_{t|0}(x_t \mid x_0)}}$的具体解析式，也就是神经网络预测score的ground truth。

- 如果$x_0$中$\mathbf{\mathbf{I}(x_0^i = j)}$不成立，由于前向过程中是不可能由一个$\mathbf{token_a}$变成$\mathbf{token_b}$的，只会逐渐变成$\mathbf{MASK}$**，所以$\frac{p_{t|0}(\hat{x}_t \mid x_0)}{p_{t|0}(x_t \mid x_0)}$此时应该是0。
- 如果$\mathbf{\mathbb{I}(x_0^i = j)}$成立，而且扩散的前向过程中每一维是独立的，所以只需要考虑$j$那一维从开始分别变成$\mathbf{MASK}$和保持不变的概率:

- 从 Lemma 1（单维度）：$p_{t|0}(x_t \mid x_0) = e^{-\bar{\sigma}(t)} \cdot \mathbb{I}(x_t = x_0) + (1 - e^{-\bar{\sigma}(t)}) \cdot \mathbb{I}(x_t = [M])$.

- 多维度扩展（Proposition 1，第 18 页，可能）：由于维度独立，联合概率是乘积。对于掩码位置 $i$，真实混凝土分数只有在 $j = x_0^i$ 时非零：$\frac{p_{t|0}(\hat{x}_t \mid x_0)}{p_{t|0}(x_t \mid x_0)} = \frac{e^{-\bar{\sigma}(t)}}{1 - e^{-\bar{\sigma}(t)}} \cdot \mathbb{I}(x_0^i = j)$。


