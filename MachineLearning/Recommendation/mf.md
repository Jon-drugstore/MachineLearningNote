# <center>Matrix Factorization</center>

<br></br>



将电影评分表格：

| Movie/User           | Alice(1) | Bob(2) | Carol(3) | Dave(4) |
|:---------------------|:---------|:-------|:---------|:--------|
| Love at last         | 5        | 5      | 0        | 0       |
| Romance for ever     | 5        | ?      | ?        | 0       |
| Cute puppies of love | ?        | 4      | 0        | ?       |
| Nonstop car chases   | 0        | 0      | 5        | 4       |
| Swords vs. karate    | 0        | 0      | 5        | ?       |

用矩阵表示：

$$
Y =
\begin{bmatrix}
5 & 5 & 0 & 0 \\
5 & ? & ? & 0 \\
? & 4 & 0 & ? \\
0 & 0 & 5 & 4 \\
0 & 0 & 5 & 0
\end{bmatrix}
$$

发现，由于用户不会对所有电影打分，所以矩阵是稀疏的。如果用预测描述这个矩阵：

$$
Predicated =
\begin{bmatrix}
(\theta^{(1)})^T x^{(1)} & (\theta^{(2)})^T x^{(1)} & \cdots & (\theta^{(n_u)})^T x^{(1)} \\
(\theta^{(1)})^T x^{(2)} & (\theta^{(2)})^T x^{(2)} & \cdots & (\theta^{(n_u)})^T x^{(2)} \\
\vdots & \vdots & \vdots & \vdots \\
(\theta^{(1)})^T x^{(n_m)} & (\theta^{(2)})^T x^{(n_m)} & \cdots & (\theta^{(n_u)})^T x^{(n_m)}
\end{bmatrix}
$$

令：

$$
X =
\begin{bmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
\vdots \\
(x^{(n_m)})^T
\end{bmatrix},

\Theta =
\begin{bmatrix}
(\theta^{(1)})^T \\
(\theta^{(2)})^T \\
\vdots \\
(\theta^{(n_u)})^T
\end{bmatrix}
$$


即$$X$$每一行描述一部电影内容，$$\Theta^T$$每一列描述用户对电影偏好程度。即，将原稀疏矩阵分解为了$$X$$和$$\Theta$$。现在预测写为：

$$
Predicated = X \Theta^T
$$

用这个方法求$$X$$和$$\Theta$$获得推荐系统需要的参数，称为**低秩矩阵分解**。该方法不仅能在编程时直接通过向量化手法获得参数，还通过矩阵分解节省了内存空间。