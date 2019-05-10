# <center>Content Based Recommendation</center>

<br></br>



## What
----
有一份用户对电影打分，`?`代表没有评价：

| Movie/User           | Alice(1) | Bob(2) | Carol(3) | Dave(4) |
|:---------------------|:---------|:-------|:---------|:--------|
| Love at last         | 5        | 5      | 0        | 0       |
| Romance for ever     | 5        | ?      | ?        | 0       |
| Cute puppies of love | ?        | 4      | 0        | ?       |
| Nonstop car chases   | 0        | 0      | 5        | 4       |
| Swords vs. karate    | 0        | 0      | 5        | ?       |


该网站对每部电影给出两个评价指数，构成二维特征向量$$x$$：

$$
\begin{align*}
x_1 = \mbox{电影的浪漫指数} \\
x_2 = \mbox{电影的动作指数}
\end{align*}
$$

| Movie/User           | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $$x_1$$ | $$x_2$$ |
|:---------------------|:---------|:-------|:---------|:--------|:--------|:--------|
| Love at last         | 5        | 5      | 0        | 0       | 0.9     | 0       |
| Romance for ever     | 5        | ?      | ?        | 0       | 1.0     | 0.01    |
| Cute puppies of love | ?        | 4      | 0        | ?       | 0.99    | 0       |
| Nonstop car chases   | 0        | 0      | 5        | 4       | 0.1     | 1.0     |
| Swords vs. karate    | 0        | 0      | 5        | ?       | 0       | 0.9     |

假设用户$$i$$对每个指数偏好程度由向量$$\theta^{(i)}$$衡量，则估计该用户对电影$$j$$的打分为：

$$
y^{(i, j)} = (\theta^{(i)})^T x^{(i)}
$$

这就是**基于内容的推荐系统**，根据商品内容判断用户偏好程度。另外，引入$$r(i,j)$$表示第$$i$$个用户是否对第$$j$$部电影打分：

$$
r(i,j) =
\begin{cases}
1, \mbox{用户 $i$ 对电影 $j$ 打过分} \\
0, \mbox{otherwise}
\end{cases}
$$

<br></br>



## 目标优化
----
为了对用户$$j$$打分作出精确预测，需要：

$$
\min_{\theta^{(j)}} = \frac{1}{2} \sum_{i:r(i,j)=1} \left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2
+ \frac{\lambda}{2} \sum_{k=1}^n (\theta_k^{(j)})^2
$$

那对所用用户$$1, 2, ... , n_u$$，需要：

$$
\min_{\theta^{(1)}, \theta^{(2)}, ..., \theta^{(n_u)}} = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n(\theta_k^{(j)})^2
$$

代价函数$$J(theta^{(1)}, \theta^{(2)}, ..., \theta^{(n_u)})$$为：

$$
J(\theta^{(1)}, \theta^{(2)}, ..., \theta^{(n_u)}) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} \left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n(\theta_k^{(j)})^2
$$

<br></br>



## 参数更新
----
使用梯度下降法更新参数：

$$
\begin{align*}
& \mbox{更新偏置（插值）：} \\
& \quad \theta_0^{(j)} = \theta_0^{(j)} - \alpha\sum_{i:r(i,j)=1}\big((\theta^{(j)})^T x^{(i)} - y^{(i,j)}\big) x_0^{(i)} \\
& \mbox{更新权重：} \\
& \quad \theta_k^{(j)} = \theta_k^{(j)} - \alpha\left(\sum_{i:r(i,j)=1}\big((\theta^{(j)})^T x^{(i)} - y^{(i,j)}\big) x_k^{(i)} + \lambda\theta_k^{(j)}\right), \quad k \neq 0
\end{align*}
$$