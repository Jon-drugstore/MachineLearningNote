# <center>Gradient Descent</center>

<br></br>



## Why
----
在特征量很大情况下，即便生成图像，人工也很难读出$$J(\theta)$$最小值，且多数情况无法可视化，故引入梯度下降（Gradient Descent），让计算机自动找出最小化代价函数时对应的$$\theta$$值。

<p align="center">
  <img src="./Images/gd6.png" width = "400"/>
</p>

<br></br>



## How
----
梯度下降思想是：
1. 开始时，随机选择一个参数组合$$( {\theta_{0}},{\theta_{1}},......,{\theta_{n}} )$$作为起始点，计算代价函数。
2. 然后寻找下一个使得代价函数下降最多的参数组合。
3. 不断迭代，直到找到一个局部最小值。

**注意，由于下降情况只考虑当前参数组合，所以无法确定当前局部最小值是否是全局最小值。不同初始参数组合，可能产生不同的局部最小值：**

<p align="center">
  <img src="./Images/gd1.jpg" width = "550"/>
</p>

**The gradient descent algorithm is to repeat until convergence:**

$$
\theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} \text{J}(\theta_0, \theta_1)
$$

where $$j = 0, 1$$ represents the feature index number.

计算时要**批量更新$$\theta$$值**，否则结果上有所出入。

<p align="center">
  <img src="./Images/gd2.png" width = "550"/>
</p>

<br></br>



## Learning Rate
----
We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter $$\alpha$$, which is called **learning rate**.

> 学习速率决定参数值变化速率，即“走多少距离”。偏导部分决定下降方向，即“下一步往哪走”。

For example, the distance between each 'star' represents a step determined by parameter $$\alpha$$. The direction is determined by the partial derivative of $$\mathit{J}(\theta_0, \theta_1)$$. Depending on where one starts on graph, one could end up at different points. 

**Hyperparameters** are the knobs that programmers tweak in machine learning algorithms. If a learning rate is too small, learning will take too long. Conversely, if a learning rate is too large, the next point will perpetually bounce haphazardly across the bottom of the well like a quantum mechanics experiment gone horribly wrong.

<p align="center">
  <img src="./Images/learning_rate1.svg" width = "400"/>
</p>

<p align="center">
  <img src="./Images/learning_rate2.svg" width = "400"/>
</p>

<br></br>



## Deep Understanding
----
Regardless of the slope's sign for $$\frac{\partial}{\partial \theta_j} \text{J}(\theta_1)$$, $$\theta_1$$ eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of $$\theta_1$$ increases and when it is positive, the value of $$\theta_1 $$decreases.

<p align="center">
  <img src="./Images/gd3.png" width = "550"/>
</p>

We should adjust parameter $$\alpha$$ to ensure that the gradient descent converges in a reasonable time. 

<p align="center">
  <img src="./Images/gd4.png" width = "550"/>
</p>

学习速率只需选定即可，不需运行时动态改变。随着斜率接近0，代价函数变化幅度会越小，直到收敛局部极小值：

<p align="center">
  <img src="./Images/gd5.png" width = "550"/>
</p>

<br></br>



## Gradient Descent For Linear Regression
----
<p align="center">
  <img src="./Images/gd7.png" width = "550"/>
</p>

将线性回归模型公式代入梯度下降公式可得：

<p align="center">
  <img src="./Images/gd8.png" width = "550"/>
</p>

当$$j=0, j=1​$$时，线性回归代价函数求导推导过程为：

$$
\begin{split}
\frac{\partial}{\partial\theta_j}J(\theta_1,\theta_2) &= \frac{\partial}{\partial\theta_j}(\frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2}) \\
&= (\frac{1}{2m} * 2\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})) * \frac{\partial}{\partial\theta_j}(h_{\theta}(x^{(i)})-y^{(i)}) \\
&= (\frac{1}{m} * \sum\limits_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})) * \frac{\partial}{\partial\theta_j} (\theta_0{x_0^{(i)}} + \theta_1{x_1^{(i)}} - y^{(i)})
\end{split}
$$

所以当$$j=0$$时：

$$
\frac{\partial}{\partial\theta_0} J(\theta) = \frac{1}{m} \sum\limits_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) * x_{0}^{(i)}
$$

所以当$$j=1$$时：

$$
\frac{\partial}{\partial\theta_1} J(\theta) = \frac{1}{m} \sum\limits_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) * x_{1}^{(i)}
$$

上文提到的梯度下降都为**批量梯度下降（Batch Gradient Descent）**，即每次计算都使用所有数据集$$\left(\sum\limits_{i=1}^{m}\right)​$$。

由于线性回归函数是凸函数，呈现碗状，即只有一个全局最优值，所以函数定会收敛到全局最小值（学习速率不可过大）。所以，线性回归函数求最小值属于凸函数优化问题。

<p align="center">
  <img src="./Images/gd9.png" width = "400"/>
</p>

<p align="center">
  <img src="./Images/gd10.png" width = "400"/>
</p>

<p align="center">
  <img src="./Images/gd11.png" width = "400"/>
</p>

<p align="center">
  <img src="./Images/gd12.png" width = "400"/>
</p>

<br></br>



## Gradients
----
The gradient of a function, denoted as follows, is the vector of partial derivatives with respect to all of the independent variables:

$$
\nabla{f}
$$

For instance, if:

$$
f(x,y) = e^{2y}\sin{(x)}
$$

then:

$$
\nabla{f(x,y)} = (\frac{\partial{f}}{\partial{x}}(x,y), \frac{\partial{f}}{\partial{y}}(x,y)) = (e^{2y}\cos{(x)}, 2e^{2y}\sin{(x)})
$$

The number of dimensions in the vector is equal to the number of variables in the formula for $$f$$; in other words, the vector falls within the domain space of the function. For instance, the graph of the following function $$f(x, y)$$:

$$
f(x, y) = 4 + (x-2)^{2} + 2y^{2}
$$

when viewed in three dimensions with $$z = f(x,y)$$ looks like a valley with a minimum at (2, 0, 4):

![](./Images/gradient1.svg)

The gradient of $$f(x, y)$$ is a two-dimensional vector that tells you in which $$(x, y)$$ direction to move for the maximum increase in height. Thus, the negative of the gradient moves you in the direction of maximum decrease in height. In other words, the negative of the gradient vector points into the valley.

In machine learning, gradients are used in gradient descent. We often have a loss function of many variables that we are trying to minimize, and we try to do this by following the negative of the gradient of the function.

Note that a gradient is a vector, so it has both of the following characteristics:
1. a direction
2. a magnitude

The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.

<br></br>



## Learning Rate
----
As noted, the gradient vector has both a direction and a magnitude. Gradient descent algorithms multiply the gradient by a scalar known as the **learning rate (step size)** to determine next point. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.

<br></br>



## Stochastic Gradient Descent 随机梯度下降
----
In gradient descent, a batch is the total number of examples you use to calculate the gradient in a single iteration. A large data set with randomly sampled examples probably contains redundant data. In fact, redundancy becomes more likely as the batch size grows. Some redundancy can be useful to smooth out noisy gradients, but enormous batches tend not to carry much more predictive value than large batches.

By choosing examples at random from data set, we could estimate a big average from a much smaller one. **Stochastic gradient descent (SGD)** takes this idea to the extreme -- it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term _stochastic_ indicates that the one example comprising each batch is chosen at random.

随机梯度下降算法首先对训练集随机洗牌，然后：

$$
\begin{aligned}
\text{Repeat until convergence} & \{ \\
& \quad \text{for}\ i = 1\ \text{to}\ m\text{:} \\
& \quad\quad \theta_{j} = \theta_{j} + \alpha(y^{(i)} - h_{\theta}(x^{(i)}))x_{j}^{(i)} \\
& \}
\end{aligned}
$$

相较于批量梯度下降，随机梯度下降每次更新$$\theta_j$$只会用当前遍历的样本。虽然外层循环仍需遍历所有样本，但往往能在样本尚未遍历完时就已收敛。因此，面临大数据集时，随机梯度下降法性能卓越。

<div style="text-align:center">
  <img src="./Images/sgd1.png" width="400"></img>
</div>

|            |           概括          |	优点                   |	缺点 |
| ---------- | ---------------------- | ----------------------- | ---- |
| 批量梯度下降 | 减小训练样本总预测代价。   | 能获得最优解，支持并行计算。 | 样本容量大时，性能显著下降。 |
| 随机梯度下降 | 减小每个训练样本预测代价。 |  训练速度快。              | 不一定获得全局最优，常出现抖动和噪音，且不能通过并行计算优化。 |

<div style="text-align:center">
  <img src="./Images/sgd7.png" width="600"></img>
</div>

<br>


### SGD Convergence
相较于批量梯度下降，随机梯度下降倾向于找到局部最优解而不是全局。因此，通常需绘制调试曲线来监控随机梯度工作过程是否正确。假定误差定义为$$cost(\theta, (x^{(i)}, y^{(i)})) = \frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2$$，则每完成1000次迭代（即遍历1000个样本），求取平均误差并绘制，得到误差随迭代次数变化曲线：

<div style="text-align:center">
  <img src="./Images/sgd2.png" width="400"></img>
</div>

另外，遇到下面曲线也不用担心，并不意味着学习率出问题，可能是平均间隔取太小：

<div style="text-align:center">
  <img src="./Images/sgd3.png" width="400"></img>
</div>

如果每进行5000次迭代才绘制，那么曲线将更加平滑：

<div style="text-align:center">
  <img src="./Images/sgd4.png" width="400"></img>
</div>

如果面临明显上升态势曲线，要考虑降低学习率$$\alpha$$：

<div style="text-align:center">
  <img src="./Images/sgd5.png" width="400"></img>
</div>

另外，学习率$$\alpha$$还可随着迭代次数优化：

$$
\alpha = \frac{constant1} {iterationNumber + constant2}
$$

随迭代次数增多，下降步调会放缓，避免出现抖动：

<div style="text-align:center">
  <img src="./Images/sgd6.png" width="400"></img>
</div>

<div style="text-align:center">
  <img src="./Images/sgd8.png" width="600"></img>
</div>

<br></br>



## Mini Batch Gradient Descent 小批量梯度下降法
----
**Mini-batch stochastic gradient descent (mini-batch SGD)** is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

小批量梯度下降是批量梯度下降和随机梯度下降折中，通过参数$$b$$指明每次迭代时，用于更新$$\theta$$的样本数。假定$$b=10,m=1000$$，小批量梯度下降工作过程如下：

$$
\begin{align*}
&\mbox{重复直到收敛：} \\
& \quad \mbox{for $i=1,11,21,...,991$：} \\
& \quad \quad \theta_j = \theta_j - \alpha\frac{1}{10}\sum_{k=i}^{i+9}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)},
\quad \mbox{for $j=0,...,n$}
\end{align*}
$$

<div style="text-align:center">
  <img src="./Images/mbgd1.png" width="600"></img>
</div>

<br></br>