# <center>SMO - Sequential minimal optimization</center>

<br></br>



## Background
----
使用核函数软间隔支持向量机的优化模型为：

$$
\begin{align*}
& \max_{\alpha} \sum_{i=1}^m\alpha^{(i)}
- \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha^{(i)} \alpha^{(j)} y^{(i)} y^{(j)} \kappa(x^{(i)}, x^{(j)}) \\
\mbox{s.t.} \quad & \sum_{i=1}^m \alpha^{(i)} y^{(i)} = 0, \\
& 0 \leq \alpha^{(i)} \leq C, \quad i=1,2,3,...,m
\end{align*}
\tag{1}
$$

式1需满足的KKT条件为：

$$
\begin{equation}
\alpha^{(i)}=0\Leftrightarrow y^{(i)}f(x^{(i)})\geq1,\\
0<\alpha^{(i)}<C\Leftrightarrow y^{(i)}f(x^{(i)})=1,\\
\alpha^{(i)}=C\Leftrightarrow y^{(i)}f(x^{(i)})\leq 1.
\end{equation}
\tag{2}
$$

在SMO（序列最小化）出现前，依赖二次规划求解工具来解决上述优化问题。这些工具需强大计算能力，实现也复杂。1998年，微软研究院的[John Platt](https://www.gitbook.com/book/yoyoyohamapi/undersercore-analysis/discussions?state=closed) 提出SMO算法将优化问题分解为易求解的若干小的优化问题。简言之，SMO仅关注$$alpha$$对和偏置$$b$$的求解更新，进而求解出权值向量$$w$$，得到决策边界（分割超平面），从而减少运算复杂度。



## 算法
----
SMO选择一对$$\alpha^{(i)}$$及$$\alpha^{(j)}$$，并固定其他参数，即将其他参数认为是常数，则式1中约束条件写为：

$$
\begin{align*}
& \alpha^{(i)} y^{(i)} + \alpha^{(j)} y^{(j)} = k, \\
& 0 \leq \alpha^{(i)} \leq C, \\
& 0 \leq \alpha^{(j)} \leq C,
\end{align*}
\tag{3}
$$

其中：

$$
k = -\sum_{k \neq i,j}\alpha^{(k)}y^{(k)} \tag{4}
$$

那么，式1优化问题可推导：

$$
\begin{align*}
& \max_{\{\alpha^{(i)}, \alpha^{(j)}\}}
(\alpha^{(i)} + \alpha^{(j)}) -
[\frac{1}{2}K_{ii}(\alpha^{(i)})^2 + \frac{1}{2}K_{jj}(\alpha^{(j)})^2 + y^{(i)}y^{(j)}K_{ij}\alpha^{(i)}\alpha^{(j)}]  \\
& \quad - [y^{(i)}\alpha^{(i)}\sum_{k=3}^my^{(k)}\alpha^{(k)}K_{ki} + y^{(j)}\alpha^{(j)}\sum_{k=3}^my^{(k)}\alpha^{(k)}K_{kj}] \\
\mbox{s.t.} \quad & \alpha^{(i)} y^{(i)} + \alpha^{(j)} y^{(j)} = -\sum_{k \neq i,j}\alpha^{(k)}y^{(k)} = k, \\
& 0 \leq \alpha^{(i)} \leq C, 0 \leq \alpha^{(j)} \leq C \\
\end{align*}
\tag{5}
$$

<div style="text-align:center">
<img src="./Images/smo1.png" width="500"></img>
</div>

- 由$$0 \leq \alpha^{(i)} \leq C, 0 \leq \alpha^{(j)} \leq C$$ 知，$$\alpha^{(i)}$$及$$\alpha^{(j)}$$取值需落在正方形中。
- 由$$\alpha^{(i)} y^{(i)} + \alpha^{(j)} y^{(j)} = k$$知，$$\alpha^{(i)}$$及$$\alpha^{(j)}$$取值需落在正方形中取值，还需落到斜线段上。

假设放缩$$\alpha^{(i)}$$取值：

$$
L \leq \alpha^{(i)} \leq H \tag{6}
$$

可确定$$\alpha^{(j)}$$上下界为：

- $$y^{(i)} \neq y^{(j)}$$时：

$$
L = max(0, \alpha^{(j)} - \alpha^{(i)}), H = min(C, C+\alpha^{(j)} - \alpha^{(i)}) \tag{7}
$$

- $$y^{(i)} = y^{(j)}$$时：

$$
L = max(0, \alpha^{(j)} + \alpha^{(i)} - C), H = min(C, \alpha^{(j)} + \alpha^{(i)}) \tag{8}
$$

将优化函数定义为：

$$
\Psi  = (\alpha^{(i)} + \alpha^{(j)}) -
[\frac{1}{2}K_{ii}(\alpha^{(i)})^2 + \frac{1}{2}K_{jj}(\alpha^{(j)})^2 + y^{(i)}y^{(j)}K_{ij}\alpha^{(i)}\alpha^{(j)}] \tag{9}
$$

由于$$\alpha^{(i)}$$与$$\alpha^{(j)}$$有线性关系，所以式9可消去$$\alpha^{(i)}$$，进而令$$\Psi$$对$$\alpha^{j}$$求二阶导，并令二阶导为0可得（中间过程略）：

$$
\alpha^{(jnew)}(2K_{ij} - K_{ii} - K_{jj}) = \alpha^{(jold)}(2K_{ij} - K_{ii} - K_{jj})
 - y^{(j)}(E^{(i)} - E^{(j)})
\tag{10}
$$

其中：

$$
\begin{align*}
E^{(i)} = f(x^{(i)}) - y^{(i)} \tag{11}
\end{align*}
$$

令：

$$
\eta = 2K_{ij} - K_{ii} - K_{jj} \tag{12}
$$

式10两边除以$$\eta$$，得$$\alpha^{(j)}$$更新式：

$$
\alpha^{(jnew)}=\alpha^{(jold)}- \frac{y^{(j)}(E^{(i)}-E^{(j)})}{\eta} \tag{13}
$$

但更新还要考虑上下界截断：

$$
\alpha^{(jnewclipped)}=
\begin{cases} H, & \mbox{if $\alpha^{(jnew)} \geq H$} \\
\alpha^{(jnew)}, & \mbox{if $L<\alpha^{(jnew)} < H$} \\
L, & \mbox{if $\alpha^{(jnew)}\leq L$}
\end{cases}
\tag{14}
$$

从而得到$$\alpha^{(i)}$$的更新：

$$
\alpha^{(inew)}=\alpha^{(iold)}+y^{(i)}y^{(j)}(\alpha^{(jold)}-\alpha^{(jnewclipped)}) \tag{15}
$$

令：

$$
\begin{align*}
b_1 &= b^{old}-E^{(i)}-y^{(i)}K_{ii}(\alpha^{(inew)}-\alpha^{(iold)})-y^{(j)}K_{ij}(\alpha^{(jnewclipped)}-\alpha^{(jold)}) \\
b_2 &=b^{old}-E^{(j)}-y^{(i)}K_{ij}(\alpha^{(inew)}-\alpha^{(old)})-y^{(j)}K_{jj}(\alpha^{(jnewclipped)}-\alpha^{(jold)})
\end{align*}
\tag{16}
$$

则$$b$$更新为：

$$
b^{new}=
\begin{cases} b_1, & \mbox{if $0<\alpha^{(inew)}<C$}; \\
b_2, & \mbox{if $0<\alpha^{(jnewclipped)}<C$};\\
\frac{b_1+b_2}{2}, & \mbox{otherwise}.
\end{cases}
\tag{17}
$$

<br></br>



## 启发式选择
----
根据[Osuna](http://webmail.svms.org/training/OsFG97.pdf)理论，如果两个拉格朗日乘子其中之一违背KKT条件，此时，每一次乘子对的选择，都能使优化目标函数减小。

- 若$$\alpha^{(i)} = 0$$，可知样本$$x^{(i)}$$不会对模型$$f(x)$$产生影响。
- 若$$\alpha^{(i)} = C$$，样本$$x^{(i)}$$不会是支持向量。
- 若$$0 < \alpha^{(i)} < C$$，则$$\alpha^{(i)}$$没落在边界上。下式满足时，$$\alpha^{(i)}$$会违反KKT条件：

$$
\mbox{$\alpha^{(i)} < C$ and $y^{(i)}f(x^{(i)}) -1 < 0$} \\
\mbox{$\alpha^{(i)} > 0$ and $y^{(i)}f(x^{(i)}) -1 > 0$}
\tag{18}
$$

式18过于严苛，因此考虑设置一个容忍区间$$[-\tau, \tau]$$，并考虑令:

$$
R^{(i)} = y^{(i)} E^{(i)} = y^{(i)} (f(x^{(i)}) - y^{(i)}) = y^{(i)}f(x^{(i)}) -1 \tag{19}
$$

可将违反KKT条件的表达式写为：

$$
\mbox{$\alpha^{(i)} < C$ and $R^{(i)} < -\tau$} \\
\mbox{$\alpha^{(i)} > 0$ and $R^{(i)} > \tau$}
\tag{20}
$$


SMO以编程的眼光，将启发式选择$$(\alpha^{(i)}, \alpha^{(j)})$$描述为了两层循环：

- 外层循环：如果当前没有$$alpha$$对的变化，意味所有$$alpha^{(i)}$$都遵从KKT条件，需在整个样本集迭代；否则，只需选择处在边界内（即$$0 < \alpha^{(i)} < C$$）、且违反KKT条件（满足式20）的$$\alpha^{(i)}$$。
- 内层循环：选出使$$|E^{(i)} - E^{(j)}|$$达到最大的$$alpha^{(j)}$$。

<br></br>



## 算法流程
----
1. 整个训练集或非边界样本中选择违反KKT条件的$$\alpha^{(i)}$$。
2. 在剩下的$$\alpha$$中，选择$$|E^{(i)} - E^{(j)}|$$达到最大的$$alpha^{(j)}$$。
3. 重复以上过程直到达到收敛精度或最大迭代次数。
