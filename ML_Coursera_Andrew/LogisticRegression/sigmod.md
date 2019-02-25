# <center>Sigmod</center>

<br></br>



How a logistic regression model can ensure output that always falls between 0 and 1. This is because sigmoid function:

$$
y = \frac{1}{1+\mathit{e}^{-z}}
$$

where
* $$y$$ is the output of logistic regression model for a particular example.
* $$z$$ is $$b + w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n}$$.
* $$w$$ is the model's learned weights and bias.
* $$x$$ is the feature values for a particular example.

If $$z$$ represents the output of linear layer of a model trained with logistic regression, then sigmoid(z) will yield a value (a probability) between 0 and 1. 

![](./Images/sigmod3.svg)

Suppose we had a logistic regression model with three features that learned the following bias and weights:
* $$b = 1$$
* $$w_{1} = 2$$
* $$w_{2} = -1$$
* $$w_{3} = 5$$

Further suppose the following feature values for a given example:
* $$x_{1} = 0$$
* $$x_{2} = 10$$
* $$x_{3} = 2$$

Therefore, the log-odds:

$$
b + w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} = 1 + 2 * 0 + (-1) * 10 + 5 * 2 = 1
$$

Consequently, the logistic regression prediction for this particular example will be 0.731% probability:

$$
y = \frac{1}{1+\mathit{e}^{-z}} = 0.731
$$

![](./Images/sigmod4.svg)

<br></br>

