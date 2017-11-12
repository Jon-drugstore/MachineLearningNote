# <center>Multiple Features and Normal Equation</center>

<br></br>



## 正规方程法
----
梯度下降算法是求目标函数最优值的一种解放。也可以直接求参数值而不用迭代，此为**正规方程法**。

定义梯度符号为$$ \nabla $$，则J的梯度表示为：

$$
\nabla_\theta J = \begin{bmatrix} \frac{\partial J}{\partial\theta_0} & \cdots & \frac{\partial J}{\partial\theta_n} \end{bmatrix}^{\rm T} \in \mathbb{R}^{n+1} \tag 8
$$

再比如，对于一个函数映射（$$ m \times n $$的矩阵到实数的映射）：

$$
\mathit{f}:\mathbb{R}^{m \times n} \to \mathbb{R}
$$

则f的梯度表示为：

$$
\nabla_A f(A) = 
\begin{bmatrix}
\frac{\partial f}{\partial A_11 } & \cdots & \frac{\partial f}{\partial A_{1n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial A_{m1}} & \cdots & \frac{\partial f}{\partial A_{mn}}
\end{bmatrix}
\tag 9$$

其中A是$$ m \times n$$矩阵。比如对于$$2 \times 2$$矩阵A，有函数f，定义为：

$$
f(A)=\frac{3}{2}A_{11}+5A^{2}_{12}+A_{21}A_{22}
$$

则得到：

$$
\nabla_a f(A) = 
\begin{bmatrix}
\frac{3}{2} & 10A_{12} \\
A_{22} & A_{21}
\end{bmatrix}
$$

对于$$n \times n$$矩阵，定义矩阵迹为：

$$
\text{trA} = \sum_{i=1}^{n}A_{ii}
\tag {10}
$$

把梯度和迹组合，得到如下性质：
1. trAB = trBA
2. trABC = trCAB = trBCA
3. trA = tr$$ \text{A}^{T} $$
4. tr(A + B) = tr(A) + tr(B)
5. traA = a * trA
6. tra = a
    其中，a是一个实数，A、B和C为$$n \times n $$矩阵
7. $$ \nabla_A \text{trAB} = \text{B}^{T} $$
8. $$ \nabla_{A^{T}}f(A) = (\nabla_Af(A))^{T} $$
9. $$ \nabla_{A}trABA^{T}C = CAB + C^{T}AB^{T} $$

<br>

### 矩阵表示目标函数
训练数据集合是$$ m \times n $$矩阵，其中m是样本个数，n为每个样本的维度。对于每个样本目标值，按照顺序排列为$$ m \times 1 $$向量。因而，数据矩阵为：

$$
X = 
\begin{bmatrix}
(x^{1})^{T} \\
\vdots \\
(x^{m})^{T}
\end{bmatrix}
\tag {11}
$$

$$
Y = \begin{bmatrix} y^{1} & \cdots & y^{m} \end{bmatrix}^{\rm T} \tag {12}
$$

因此，可以得到：

$$
X\theta - Y = 
\begin{bmatrix}
(x^{1})^{T}\theta \\
\vdots \\
(x^{m})^{T}\theta
\end{bmatrix}
-
\begin{bmatrix}
y^{1} \\
\vdots \\
y^{m}
\end{bmatrix}
=
\begin{bmatrix}
h_{\theta}(x^{1}) - y^{1} \\
\vdots \\
h_{\theta}(x^{m}) - y ^{m}
\end{bmatrix}
\tag {13}
$$

所以，得到目标函数J向量表达为：

$$
J(\theta) = \frac{1}{2} (X\theta - Y)^{T}(X\theta - Y) = \frac{1}{2} \sum^{m}{i=1}(h_{\theta}(x^{i}) - y^{i})^{2}
\tag {14}
$$

可以得到计算J的梯度公式推导：

<p align="center">
  <img src="./Images/week1_2.png" width = "550"/>
</p>

公式15中：
1. 第一行展开；
2. 第二行应用性质6；
3. 第三行应用性质4，且$$ Y^{T}Y $$为常数；
4. 第四行应用性质3；
5. 第五行应用性质1；
6. 第六行的I是单位矩阵，并应用性质9和7.

得到结果后，令导数为0，得到：

$$
X^{T}X\theta = X^{T}Y \to \theta = (X^{T}X)^{-1}X^{T}Y
\tag {16}
$$

从而，求出了参数，这种方法称为正规方程组。