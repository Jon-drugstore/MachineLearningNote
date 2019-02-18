# <center>Logistic Regression</center>

<br></br>



## Hypothesis Representation
----
为使$$h_\theta(x) \in(0, 1)$$，引入逻辑回归模型，定义假设函数

$$
h_\theta(x)=g(z)=g(\theta^{T}x)
$$

其中，$$z=\theta^{T}x$$是分类边界，且$$g(z)=\frac{1}{1+e^{-z}}$$。对比线性回归函数$$h_\theta(x)=\theta^{T}x$$，$$g$$表示逻辑函数，复合起来，称为逻辑回归函数。

**逻辑函数是S形函数，将所有实数映射到$$(0, 1)$$范围。**$$g(z)$$是Sigmoid Function，亦称Logic Function： 

<p align="center">
  <img src="./Images/sigmod1.jpg" width = "400"/>
</p>

可看到，预测函数$$h_\theta(x)$$被限制在区间$$(0, 1)$$，且Sigmod是一个很好的阀值函数，阀值为0.5。

应用Sigmoid函数，则逻辑回归模型：$$h_{\theta}(x)=g(\theta^Tx) =\frac{1}{1+e^{-\theta^Tx}}$$。模型中，$$h_\theta(x)$$作用是根据输入$$x$$及参数$$\theta$$，计算输出$$y=1$$的可能性（estimated probability）。概率学表示为：

$$
h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \\
\text{其中} P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1
$$

以肿瘤诊断为例，$$h_\theta(x)=0.7$$表示$$70\%$$概率恶性肿瘤。

<br></br>



## Decision Boundary
----
决策边界是用来划清界限的边界。边界形态可以是点、线或平面。决策边界是预测函数$$h_{\theta}(x)$$的属性，而不是训练集属性。因为能作出划清类间界限的只有$$h_{\theta}(x)$$，而训练集只用来训练和调节参数。

<br>


### 线性决策边界
逻辑回归有假设函数$$h_\theta ( x )=g(z)=g(\theta^{T}x )$$。规定$$0.5$$为阈值：

$$
h_\theta(x) \geq 0.5 \Rightarrow y = 1 \\
h_\theta(x) < 0.5 \Rightarrow y = 0
$$

![Sigmoid Function](./Images/sigmod2.png)

观察Sigmoid函数图像得当$$g(z) \geq 0.5$$，有$$z \geq 0$$，即$$\theta^Tx \geq 0$$。

同线性回归不同点在于：

$$
z \to +\infty, e^{-\infty} \to 0 \Rightarrow g(z)=1 \\
z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0
$$

线性预测函数$${h_\theta}( x )=g( {\theta_0}+{\theta_1}{x_1}+{\theta_{2}}{x_{2}})​$$是下图模型假设函数：

<p align="center">
  <img src="./Images/lr1.png" width = "400"/>
</p>

要进行分类，那么只要$${\theta_0}+{\theta_1}{x_1}+{\theta_{2}}{x_{2}}\geq0$$时，就预测$$y = 1$$，即预测为正向类。

如果取$$\theta = \begin{bmatrix} -3\\1\\1\end{bmatrix}$$，有$$z = -3+{x_1}+{x_2}$$。当$$z \geq 0$$ 即$${x_1}+{x_2} \geq 3$$时，易绘制图中品红色直线，即**决策边界**，为正向类（红叉数据）给出$$y=1$$分类预测结果。

<br>


### 非线性决策边界
为拟合下图数据，建模多项式假设函数：

$$
{h_\theta}( x )=g( {\theta_0}+{\theta_1}{x_1}+{\theta_{2}}{x_{2}}+{\theta_{3}}x_{1}^{2}+{\theta_{4}}x_{2}^{2} )
$$

取$$\theta = \begin{bmatrix} -1\\0\\0\\1\\1\end{bmatrix}$$，决策边界对应一个在原点处的单位圆（$${x_1}^2+{x_2}^2 = 1$$）。如此给出分类结果，如图中品红色曲线：

<p align="center">
  <img src="./Images/lr2.png" width = "400"/>
</p>

当然，通过一些更为复杂的多项式，还能拟合那些图像显得非常怪异的数据，使得决策边界形似碗状、爱心状等等。

<br></br>



## Cost Function
----
如果用线性回归代价函数：$$J(\theta)=\frac{1}{2m}\sum\limits_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2}$$，其中$$h_\theta(x) = g(\theta^{T}x)$$，可绘制关于$$J(\theta)$$图像如下左图。回忆线性回归平方损失函数，其是一个二次凸函数（碗状）。二次凸函数只有一个局部最小点，即全局最小点。下左图有许多局部最小点，这样梯度下降算法无法确定收敛点是全局最优。

<p align="center">
  <img src="./Images/cost1.png" width = "400"/>
</p>

对逻辑回归，更换平方损失函数为**对数损失函数**，可由统计学最大似然估计法推出代价函数$$J(\theta)$$：

$$
J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\
\mathrm{Cost}(h_\theta(x),y) =
  \begin{cases}
    -\log(h_\theta(x))       & \quad \text{if } y=1 \\
    -\log(1-h_\theta(x))  & \quad \text{if } y=0
  \end{cases}
$$

有关$$J(\theta)$$cos图像如下：

<p align="center">
  <img src="./Images/cost2.png" width = "400"/>
</p>

如左图，当训练集结果为$$y=1$$（正样本）时，随着假设函数趋向$$1$$，代价函数趋于$$0$$，意味拟合程度好。如果假设函数趋于0，则给出一个很高代价，拟合程度差。算法会根据其纠正$$\theta$$值。右图$$y=0$$同理。

<br></br>



## Simplified Cost Function and Gradient Descent
----
对二元分类问题，把代价函数简化为一个函数：

$$
\mathrm{Cost}(h_{\theta}(x), y)=-y \times log(h_{\theta}(x)) - (1-y) \times log(1-h_{\theta}(x))
$$

当$$y = 0$$，左边式子整体为0。当$$y = 1$$，则$$1-y=0$$，右边式子整体为0，和上面分段函数一样，而一个式子计算起来更方便。

$$
J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^{m} [y^{(i)}log(h_{\theta}(x^{(i)})) + (1 - y^{(i)})log(1 - h_{\theta}(x^{(i)}))]
$$

向量化实现：

$$
h = g(X\theta) \\
J(\theta) = \frac{1}{m} (-y^{T}log(h)-(1-y)^{T}log(1-h))
$$

为最优化$$\theta$$，仍使用梯度下降法，算法同线性回归中一致：

$$
\text{repeat until convergence: }\lbrace \\
\theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial{\theta_{j}}}J({\theta}) \\
\rbrace
$$

解出偏导得：

$$
\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0,1...n}\newline \rbrace\end{align*}
$$

虽形式上梯度下降算法同线性回归一样，但其中假设函不同，即$$h_\theta(x) = g(\theta^{T}x)$$，不过求导后的结果也相同。

向量化实现：$$\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - y)$$

<br>


### 代价函数求导过程
首先，

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m [y^{(i)}log (h_\theta (x^{(i)})) + (1 - y^{(i)})log (1 - h_\theta(x^{(i)}))]
$$

令$$f(\theta) = y^{(i)}log(h_{\theta}(x^{(i)})) + ( 1-y^{(i)})log(1-h_{\theta}(x^{(i)}))$$。由于$$h_\theta(x) = g(z)$$，$$g(z) = \frac{1}{1+e^{(-z)}}$$，则

$$
\begin{split}
f(\theta) &= y^{(i)}log(\frac{1}{1+e^{-z}})+(1-y^{(i)})log(1-\frac{1}{1+e^{-z}}) \\
&= -y^{(i)} log(1 + e^{-z}) - (1-y^{(i)})log(1+e^{z})
\end{split}
$$

因为$$z=\theta^Tx^{(i)}$$，对$$\theta_j$$求偏导没有$$\theta_j$$的项，求偏导为0，都消去得：

$$
\frac{\partial{z}}{\partial{\theta_{j}}} = \frac{\partial}{\partial{\theta_{j}}}(\theta^{T}x^{(i)})=x^{(i)}_{j}
$$

所以有：

$$
\begin{split}
\frac{\partial}{\partial{\theta_{j}}}f(\theta) &= \frac{\partial}{\partial{\theta_{j}}}[-y^{(i)}log(1+e^{-z})-(1-y^{(i)})log(1+e^{z})] \\
&= -y^{(i)} \frac{\frac{\partial}{\partial{\theta_{j}}}(-z)e^{-z}}{1+e^{-z}} - (1-y^{(i)}) \frac{\frac{\partial}{\partial{\theta_{j}}}(z)e^{z}}{1+e^{z}} \\
&= -y^{(i)} \frac{-x^{(i)}_{j}e^{-z}}{1+e^{-z}} - (1-y^{(i)})\frac{x^{(i)}_{j}}{1+e^{-z}} \\
&= (y^{(i)} \frac{e^{-z}}{1+e^{-z}} - (1-y^{(i)}) \frac{1}{1+e^{-z}})x^{(i)}_{j} \\
&= (\frac{y^{(i)}(e^{-z}+1)-1}{1+e^{-z}})x^{(i)}_{j} \\
&=(y^{(i)} - \frac{1}{1+e^{-z}})x_{j}^{(i)} \\
&=(y^{(i)} - h_{\theta}(x^{(i)}))x_{j}^{(i)} \\
&=-(h_{\theta}(x^{(i)}) - y^{(i)})x_{j}^{(i)}
\end{split}
$$

则可得代价函数导数：

$$
\frac{\partial}{\partial{\theta_{j}}} J(\theta) = -\frac{1}{m} \sum\limits_{i=1}^{m} {\frac{\partial}{\partial{\theta_{j}}}f(\theta)} = \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_{j}^{(i)}
$$
