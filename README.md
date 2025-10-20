
<p align="right">
  <a href="./README_CN.md" title="简体中文"><img src="https://img.shields.io/badge/语言-简体中文-blue?logo=googletranslate" alt="简体中文"></a>
</p>

# MGUP: A Momentum-Gradient Greedy Alignment Update Policy for Stochastic Optimization


Authors: Da Chang, Ganzhao Yuan

Our article is accepted as Spotlight by NeurIPS 2025. You can find this in [NeurIPS2025](https://neurips.cc/virtual/2025/poster/117868) Here it is.

## Core Algorithm  
Our central claim is the **MGUP** strategy. A safeguard mechanism controls a threshold: by ranking the products $m_{t,i}\cdot g_{t,i}$, the optimizer grants *larger* step-sizes to parameters whose momentum and stochastic-gradient directions are highly aligned, while the remaining parameters receive *non-zero* but small steps. Ranking prevents the extreme case in which only a tiny fraction of coordinates align; giving aligned coordinates larger steps is a greedy acceleration. Crucially, keeping the unaligned coordinates *non-zero* is essential for Adam—naïvely dropping them (the [Cautious](https://github.com/kyleliang919/C-Optim) trick) can make Adam diverge. We rigorously prove that MGUP-Adam converges in the stochastic non-convex setting. The proposed method can be viewed as an intra-layer learning rate adjustment strategy.

In practice you can use **Cautious-MGUP** to avoid the expensive Top-K sort on very large models:

$$
\phi_{t,i}= 
\begin{cases}
\alpha & \text{if }\mathbf m_{t,i}\cdot\mathbf g_{t,i}>0\\[4pt]
\gamma & \text{if }\mathbf m_{t,i}\cdot\mathbf g_{t,i}\le 0
\end{cases}
$$

**Note 1:** Our theory is confined to Adam; whether Lion, Muon, etc. can safely adopt the Cautious trick without losing convergence remains open.

**Note 2:** The stepsize increase factor $\alpha$ and decrease factor $\gamma$ are currently adjusted heuristically and **will not be useful in all cases**. We use the notation $\alpha=1/\tau,\gamma=\tau$. The step size cannot be scaled up indefinitely, so in practice it has to be scaled down when the scaling step size by a factor of $1/\tau$ is too large resulting in performance degradation. **Large-scale models** exhibit a heightened sensitivity to the learning rate. Consequently, it is a relatively reasonable practice to set the learning rate scaling parameter $\alpha$ within the range $[1.0, 1.5]$ to prevent excessively large steps from leading to suboptimal updates, and to set $\gamma$ within the range $[0.5, 1.0]$. Specifically, **when utilizing the MGUP optimizer, if the learning rate of the base optimizer has already been thoroughly tuned**, the selection of $\alpha$ and $\gamma$ should be made with **due caution**.

## Usage  
Set the alignment threshold via `mask_ratio` and scale the steps of coordinates *not* in the top-K via `gamma`; $0.1–0.5$ usually works well.

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
            ### MGUP parameters
            mask_ratio=0.5,
            alpha=2.0
            gamma=0.1,
            ###############
            no_deprecation_warning: bool = False,
    ):
```

```python
from MGUP.MGUP_AdamW import AdamW as mg_adamw
from MGUP.MGUP_AdamW import CMGUP_AdamW as cmg_adamw
```

## Some experiments

The following experiments detail the configurations and results of the training processes:

**Experiment 1: Single RTX-4090 GPU**

  * Model Architecture: Qwen2.5-150M
  * Training Dataset: Wikitext-103
  * Number of Training Epochs: 5
  * Batch Size: 160

The learning rate schedule, as well as the training and validation loss curves, are presented below. Figure 1 illustrates the learning rate schedule, Figure 2 depicts the training loss curve, and Figure 3 shows the validation loss curve.

<p align="center">
<table>
  <tr>
    <td><img src="./img/Learning Rate Schedule.png" alt="Figure 1: Learning Rate Schedule" width="300"/></td>
    <td><img src="./img/Qwen2.5-150M WikiText-103 Pre-training Loss Curves-AdamW-Type.png" alt="Figure 2: Training Loss" width="300"/></td>
    <td><img src="./img/Qwen2.5-150M WikiText-103 Pre-training Val Loss Curves-AdamW-Type.png" alt="Figure 3: Validation Loss" width="300"/></td>
  </tr>
</table>
</p>

**Experiment 2: Single ASCEND-910C NPU**

  * Model Architecture: LLaMA2-130M
  * Training Dataset: C4
  * Number of Training Steps: 10,000
  * Batch Size: 512

The learning rate schedule, as well as the training and validation loss curves, are presented below. Figure 1 illustrates the learning rate schedule, Figure 2 depicts the training loss curve, and Figure 3 shows the validation loss curve.

<p align="center">
<table>
  <tr>
    <td><img src="./img/LLaMA2 Learning Rate Schedule.png" alt="Figure 1: Learning Rate Schedule" width="300"/></td>
    <td><img src="./img/LLaMA2-130M C4 Pre-training Loss Curves-AdamW-Type.png" alt="Figure 2: Training Loss" width="300"/></td>
    <td><img src="./img/LLaMA2-130M C4 Pre-training Val Loss Curves-AdamW-Type.png" alt="Figure 3: Validation Loss" width="300"/></td>
  </tr>
</table>
</p>