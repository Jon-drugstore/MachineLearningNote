# <center>Advanced Optimization</center>

<br></br>



## What
----
运行梯度下降算法，其能最小化代价函数$$J(\theta)$$并得出$$\theta$$最优值。使用梯度下降算法时，如果不需观察代价函数收敛情况，则直接计算$$J(\theta)$$导数项即可，而不需计算$$J(\theta)$$值。

编写代码给出代价函数及其偏导数然后传入梯度下降算法中，接下来算法会最小化代价函数给出参数最优解，这类算法称为**最优化算法（Optimization Algorithms）**。

一些最优化算法：
- 梯度下降法 Gradient Descent
- 共轭梯度算法 Conjugate Gradient
- 牛顿法和拟牛顿法 Newton's Method & Quasi-Newton Methods
  - DFP算法
  - 局部优化法 BFGS 
  - 有限内存局部优化法 L-BFGS
- 拉格朗日乘数法 Lagrange Multiplier

<br></br>



## Example
----
假设$$J(\theta) = (\theta_1-5)^2 + (\theta_2-5)^2$$，要求参数$$\theta=\begin{bmatrix} \theta_1\\\theta_2\end{bmatrix}$$最优值。

Octave求解最优化问题代码实例：

1. 创建函数返回代价函数及其偏导数：

    ```matlab
function [jVal, gradient] = costFunction(theta)
  % code to compute J(theta)
  jVal=(theta(1)-5)^2+(theta(2)-5)^2;

  % code to compute derivative of J(theta)
  gradient=zeros(2,1);
  
  gradient(1)=2*(theta(1)-5);
  gradient(2)=2*(theta(2)-5);
end
    ```

2. 将`costFunction`函数及参数传入最优化函数`fminunc`求最优化问题：

    ```matlab
% 'GradObj', 'on'：启用梯度目标参数（则需要将梯度传入算法）。
% 'MaxIter', 100：最大迭代次数为100次。
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
    % @xxx：函数指针。
    % optTheta：最优化得到的参数向量。
    % functionVal：引用函数最后一次的返回值。
    % exitFlag：标记代价函数是否收敛。
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
    ```

3. 返回结果：

```matlab
optTheta =
     5
     5

functionVal = 0

exitFlag = 1
```
