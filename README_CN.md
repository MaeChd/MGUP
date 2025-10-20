<p align="right">
  <a href="./README.md" title="English"><img src="https://img.shields.io/badge/Language-English-blue?logo=github" alt="English"></a>
</p>

# MGUP: A Momentum-Gradient Greedy Alignment Update Policy for Stochastic Optimization

Authors: Da Chang, Ganzhao Yuan

Our article is accepted as Spotlight by NeurIPS 2025. You can find this in [NeurIPS2025](https://neurips.cc/virtual/2025/poster/117868) Here it is.

## 核心算法
我们的核心主张在于MGUP策略，采用safeguard机制通过控制阈值，对$m_{t,i} \cdot g_{t,i}$进行排序，在优化器中对动量与随机梯度方向一致程度大的元素执行更大步长，而其余的元素则执行非零的小步长。采用排序是为了防止动量与梯度方向极少量对齐的极端情况发生，而一致的元素采用更大的步长则是贪心策略。而对于不一致的元素采用非零的小步长，这在Adam中是重要的，因为简单执行[Cautious](https://github.com/kyleliang919/C-Optim)策略可能导致Adam不收敛。我们在论文中严格证明了在随机非凸优化设置下MGUP-Adam的收敛性。该方法可以被视为一种层内学习率调整策略。




实践中可以尝试Cautious-MGUP,避免大规模参数下成本较大的TopK排序:
$$\phi_{t,i} = \begin{cases} 
\alpha & \text{if } \mathbf{m}_{t,i} \cdot \mathbf{g}_{t,i} > 0 \\ 
\gamma & \text{if } \mathbf{m}_{t,i} \cdot \mathbf{g}_{t,i} \le 0 
\end{cases}$$

Note 1: 我们的理论分析限于Adam，因此对于Lion和Muon等算法的是否存在使用Cautious策略而导致不收敛是一个开放的问题。

Note 2: 步长的增大倍率$\alpha$与缩小倍率$\gamma$目前的调整方式是启发式的，`因此不会在任何情况下都实用`。我们在文中采用的方式是$\alpha=1/\tau,\gamma=\tau$。步长并不能无限放大，因此在实践中当放大步长的倍数$1/\tau$过大导致性能下降时，必须降低倍数。**大规模模型**对学习率表现出更高的敏感性，因此将学习率缩放参数 $\alpha$ 设定在 $[1.0, 1.5]$ 范围内，避免过大的步长导致次优更新，将 $\gamma$ 设定在 $[0.5, 1.0]$ 范围内是比较合理的选择。特别地，**在使用MGUP优化器时，若基础优化器的学习率已被充分调整**，则应审慎考虑 $\alpha$ 和 $\gamma$ 的取值。



```python
class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            ### MGUP parameter 
            mask_ratio=0.5,
            alpha=2.0,
            gamma=0.1,
            ###############
            no_deprecation_warning: bool = False,
    ):
```

```python
from MGUP.MGUP_AdamW import AdamW as mg_adamw
from MGUP.MGUP_AdamW import CMGUP_AdamW as cmg_adamw
```

## 部分结果



以下训练实验的配置与结果如下：

**实验一：单卡RTX-4090**

  * 模型架构：Qwen2.5-150M
  * 训练数据集：Wikitext-103
  * 训练轮数：5轮
  * 批量大小（Batch size）：160
  * 学习率调整策略及训练、验证损失曲线如下，其中图1展示学习率调度，图2呈现训练损失变化，图3展示验证损失情况 。

<p align="center">
<table>
  <tr>
    <td><img src="./img/Learning Rate Schedule.png" alt="图1 LR" width="300"/></td>
    <td><img src="./img/Qwen2.5-150M WikiText-103 Pre-training Loss Curves-AdamW-Type.png" alt="图2 Training Loss" width="300"/></td>
    <td><img src="./img/Qwen2.5-150M WikiText-103 Pre-training Val Loss Curves-AdamW-Type.png" alt="图3 Val Loss" width="300"/></td>
  </tr>
</table>
</p>

**实验二：单卡ASCEND-910C**

  * 模型架构：LLaMA2-130M
  * 训练数据集：C4
  * 训练步数（Steps）：10000步
  * 批量大小（Batch size）：512
  * 学习率调整策略及训练、验证损失曲线如下，其中图1展示学习率调度，图2呈现训练损失变化，图3展示验证损失情况 。

<p align="center">
<table>
  <tr>
    <td><img src="./img/LLaMA2 Learning Rate Schedule.png" alt="图1 LR" width="300"/></td>
    <td><img src="./img/LLaMA2-130M C4 Pre-training Loss Curves-AdamW-Type.png" alt="图2 Training Loss" width="300"/></td>
    <td><img src="./img/LLaMA2-130M C4 Pre-training Val Loss Curves-AdamW-Type.png" alt="图3 Val Loss" width="300"/></td>
  </tr>
</table>
</p>




