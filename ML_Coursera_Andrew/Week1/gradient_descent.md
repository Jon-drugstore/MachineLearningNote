# <center>Gradient Descent</center>

<br></br>



Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields $$ \theta_0 $$ and $$ \theta_1 $$. We are not graphing _x_ and _y_ itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters. We put $$ \theta_0 $$ on the _x_ axis and $$ \theta_1 $$ on the _y_ axis, with the cost function on the vertical _z_ axis. The points on our graph will be the result of the cost function using our hypothesis with those specific $$ \theta $$ parameters.

<p align="center">
  <img src="./Images/gd1.png" width = "550"/>
</p>

<br>

We will know that we have succeeded when our cost function is at the very bottom of the pits in graph.

The way we do this is by taking the derivative (the tangential line to a function) of cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter $$ \alpha $$, which is called **learning rate**.

For example, the distance between each 'star' represents a step determined by parameter $$ \alpha $$. The direction is determined by the partial derivative of $$ \mathit{J}(\theta_0, \theta_1) $$. Depending on where one starts on graph, one could end up at different points. 

**The gradient descent algorithm is to repeat until convergence:**

$$
\theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial \theta_{j}} \text{J}(\theta_0, \theta_1)
$$

where $$ j = 0, 1 $$ represents the feature index number.

At each iteration _j_, one should simultaneously update the parameters $$ \theta_1 $$, $$ \theta_2 $$, ..., $$ \theta_n $$. Updating a specific parameter prior to calculating another one on the j(th) iteration would yield to a wrong implementation. 

<p align="center">
  <img src="./Images/gd2.png" width = "550"/>
</p>

<br>

Regardless of the slope's sign for $$ \frac{\partial}{\partial \theta_j} \text{J}(\theta_1) $$, $$ \theta_1 $$ eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of $$ \theta_1 $$ increases and when it is positive, the value of $$ \theta_1 $$ decreases.

<p align="center">
  <img src="./Images/gd3.png" width = "550"/>
</p>

<br>

We should adjust parameter $$ \alpha $$ to ensure that the gradient descent converges in a reasonable time. 

<p align="center">
  <img src="./Images/gd4.png" width = "550"/>
</p>

<br>

The intuition behind the convergence is that $$ \frac{\partial}{\partial \theta_j} \mathrm{J}(\theta_1) $$ approaches $$ 0 $$ as we approach the bottom of convex function. At the minimum, the derivative will always be $$ 0 $$ and thus we get: 

$$
\theta_1 := \theta_1 - \alpha \times 0
$$

<p align="center">
  <img src="./Images/gd5.png" width = "550"/>
</p>

<br>

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute actual cost function and actual hypothesis function and modify the equation to (repeat until convergence):

$$
\theta_0 := \theta_0 -\alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta{x_i}-y_i)
$$

$$
\theta_1 := \theta_1 -\alpha \frac{1}{m} \sum_{i=1}^{m}((h_\theta{x_i}-y_i)x_i)
$$


where _m_ is the size of the training set, $$ \theta_0 $$ is a constant that will be changing simultaneously with $$ \theta_1 $$ and $$ x_i $$, $$ y_i $$ are values of the given training set. 

$$
\frac{\partial}{\partial \theta_j}\text{J}(\theta) = \frac{\partial}{\partial \theta_j}\frac{1}{2}(\text{h}_\theta(x)-y)^2 = (\text{h}_\theta(x)-y)x_j
$$