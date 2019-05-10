# <center>Mean Normalization</center>

<br></br>



假定新注册用户`Eve(5)`，他还没对任何电影作出评价：

| Movie/User           | Alice(1) | Bob(2) | Carol(3) | Dave(4) | Eve(5) |
|:---------------------|:---------|:-------|:---------|:--------|:-------|
| Love at last         | 5        | 5      | 0        | 0       | ?      |
| Romance for ever     | 5        | ?      | ?        | 0       | ?      |
| Cute puppies of love | ?        | 4      | 0        | ?       | ?      |
| Nonstop car chases   | 0        | 0      | 5        | 4       | ?      |
| Swords vs. karate    | 0        | 0      | 5        | ?       | ?      |

$$
Y =
\begin{bmatrix}
5 & 5 & 0 & 0 & ?\\
5 & ? & ? & 0 & ?\\
? & 4 & 0 & ? & ?\\
0 & 0 & 5 & 4 & ?\\
0 & 0 & 5 & 0 & ?
\end{bmatrix}
$$

则`Eve(5)`对电影偏好应当被参数$$\theta^{(5)}$$评估，注意到最小化代价函数过程：

$$
\min_{x^{(i)},...,x^{(n_m)} ; \theta^{(1)}, ..., \theta^{(n_u)}}
\frac{1}{2} \sum_{(i,j):r(i,j)=1}  \left( (\theta^{(j)})^T x^{(i)} - y^{(i,j)}\right) ^2
+ \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n (x_k^{(i)})^2
+ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n(\theta_k^{(j)})^2
$$

由于该用户没对任何电影作出评价，$$\theta^{(5)}$$能影响上式的项只有：

$$
\frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n(\theta_k^{(j)})^2
$$

为最小化该式，令$$\theta^{(5)} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$，从而`Eve(5)`对任何电影评价会被预测为：

$$
y(i, 5) = (\theta^{(5)})^T x^{(i)}= 0
$$

显然，这是“不负责任”预测，系统会认为`Eve`对任何电影都不感冒。为解决，先求各电影平均得分$$\mu$$：

$$
\mu =
\begin{pmatrix}
2.5 \\
2.5 \\
2 \\
2.25 \\
1.25
\end{pmatrix}
$$

并求$$Y-\mu$$，对$$Y$$进行均值标准化：

$$
Y - \mu =
\begin{bmatrix}
2.5 & 2.5 & -2.5 & -2.5 & ? \\
2.5 & ? & ? & -2.5 & ? \\
? & -2 & -2 & ? & ? \\
-2.25 & -2.25 & 2.75 & 1.75 & ? \\
-1.25 & -1.25 & 3.75 & -1.25 & ?
\end{bmatrix}
$$

对用户$$j$$，他对电影$$i$$评分就为：

$$
y(i, j) = (\theta^{(i)})^T x^{(j)} + \mu_i
$$

那么Eve对电影评分为：

$$
y(i, 5) = (\theta^{(5)})^T x^{(i)} + \mu_i = \mu_i
$$

即，系统在用户未给出评价时，默认该用户对电影的评价与其他用户平均评价一致。