# <center>Cost Function</center>



<br></br>

神经网络代价函数公式：

$$
h_\Theta(x) = a^{(L)} = g(\Theta^{(L-1)}a^{(L-1)}) = g(z^{(L)})
$$

$$
\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}
$$

其中：
* $$L$$：神经网络总层数。
* $$s_l$$：第$$l$$层激活单元的数量（不含偏置单元）。
* $$h_\Theta(x)_k$$：第$$k$$个分类（$$k^{th}$$）的概率，即$$P(y=k | x ; \Theta)$$。
* $$K$$：输出层的输出单元数量。
* $$y_k^{(i)}$$：第$$i$$个训练样本的第$$k$$个分量值。
* $$y$$：$$K$$维向量。

1. 左半部分是为求解$$K$$分类问题，即公式对每个样本特征运行$$K$$次，并依次给出分为第$$K$$类的概率，$$h_\Theta(x)\in \mathbb{R}^{K}, y \in \mathbb{R}^{K}$$。
2. 右半部分每一层有多维矩阵$$\Theta^{(l)}\in \mathbb{R}^{(s_l + 1)\times s_{l+1}}$$。从左到右看三次求和式$$\sum\limits_{l=1}^{L-1}\sum\limits_{i=1}^{s_l}\sum\limits_{j=1}^{s_{l+1}}$$ ，是对每一层间多维矩权重$$\Theta^{(l)}$$依次平方后求取其除了偏置权重部分的和值，并循环累加得结果。其中，$$\mathbb{R}^{m}$$为$$m$$维向量，$$\mathbb{R}^{m\times n}$$为$$m \times n$$维矩阵。

可发现，神经网络代价函数与如下的逻辑回归代价函数相似：

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2
$$

可见，神经网络背后思想是和逻辑回归一样。但由于计算复杂，神经网络代价函数$$J(\Theta)$$是一个非凸（non-convex）函数。