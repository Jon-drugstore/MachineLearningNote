# <center>Artificial Neural Networks</center>



<br></br>

<p align="center">
  <img src="./Images/ann19.jpg" width = "900"/>
</p>

<br></br>



## Methdology
----
应用神经网络有步骤：

1. 神经网络建模
   - 选取特征，确定特征向量$$x$$维度，即输入单元数量。
   - 鉴别分类，确定预测向量$$h_\Theta(x)$$维度，即输出单元数量。
   - 确定隐藏层有几层及每层隐藏层有多少个隐藏单元。

   > 默认情况下，隐藏层至少有一层。层数越多意味着效果越好，计算量越大。

2. 训练神经网络

   1. 随机初始化初始权重矩阵。
   2. 应用前向传播计算初始预测。
   3. 计算代价函数$$J(\Theta)$$。
   4. 应用后向传播计算$$J(\Theta)$$偏导数。
   5. 使用梯度检验检查算法正确性。用完就禁用它。
   6. 丢给最优化函数最小化代价函数
      > 由于神经网络的代价函数非凸，最优化时不一定收敛在全局最小值。高级最优化函数确保收敛在某个**局部**最小值。

![](./Images/quiz5.png)

<br></br>



## Types of Neural Networks
----
1. Feedforward
    
    **The information moves in only one directio, forward. There are no cycles or loops in the network.** Single-layer Perceptron, Multi-layer perceptron, and Convolutional Neural Network (CNN) are this type.

    ![](./Images/ann21.jpg)

2. Feedback
    
    In Feedback, loops are allowed. They are used in content addressable memories. Recurrent Neural Network (RNN) is this type.

    ![](./Images/ann22.jpg)

<br></br>



## Neuron 神经元
----
神经元模型是一个包含输入，输出与计算功能的模型。

<p align="center">
  <img src="./Images/ann1.jpg" width = "600"/>
</p>

<center><i>典型的神经元模型：3个输入，1个输出，及2个计算。</i></center>

<br>

神经网络训练算法就是让权重值调到最佳，使得整个网络预测效果最好。

使用_a_表示输入，_w_表示权值。连接的有向箭头可理解为在初端，传递的信号大小是_a_，端中间有加权参数_w_，加权后信号变成$$a * w$$。因此在连接末端，信号大小变成$$a*w$$。如果将神经元图中所有变量用符号表示，且写出输出计算公式：

<p align="center">
  <img src="./Images/ann2.jpg" width = "600"/>
</p>

<center><i>神经元计算</i></center>

<br>

对神经元模型图进行扩展。首先将sum函数与sgn函数合并，代表神经元内部计算。其次，一个神经元可以引出多个输出的有向箭头，但值是一样的。神经元可看作计算与存储单元。计算是神经元对其的输入进行计算功能。存储是神经元暂存计算结果，并传递下一层。

> 描述网络中某个神经元时，更多用“单元”（unit）指代，有时也会用“节点”（node）表达。

其中，**函数_f_是非线性的且称作激活函数（Activation Function），把神经元输出变成非线性。**

激活函数有：
* Sigmoid: takes a real-valued input and squashes it to range between 0 and 1.
$$
\sigma(x) = \frac{1}{1 + \mathrm{e}^{-x}}
$$

* tanh: takes a real-valued input and squashes it to the range [-1, 1].
$$
tanh(x) = 2 \sigma(2x) - 1
$$

* ReLU (Rectified Linear Unit): takes a real-valued input and thresholds it at zero (replaces negative values with zero).
$$
f(x) = max(0, x)
$$

<p align="center">
  <img src="./Images/ann4.png" width = "700"/>
</p>

<center><i>Different Activation Functions</i></center>

<br></br>



## 单层神经网络 Single-layer Perceptron
----
假要预测的不是一个值，是一个向量，如$$[2,3]$$，那么输出层增加一个输出单元。

<p align="center">
  <img src="./Images/ann5.jpg" width = "400"/>
</p>

<center><i>单层神经网络</i></center>

<br>

可看到，$$z_1$$计算跟原先_z_没有区别。$$z_2$$计算中除三个新的权值$$w_4$$，$$w_5$$和$$w_6$$外，其他与$$z_1$$一样。改用二维下标，用$$w_{x,y}$$表达权值。_x_代表后一层神经元序号，_y_代表前一层神经元的序号。例如，$$$w_{1,2}$$代表后一层第1个神经元与前一层第2个神经元的连接的权值。

<p align="center">
  <img src="./Images/ann6.jpg" width = "400"/>
</p>

<center><i>单层神经网络（扩展）</i></center>

<br>

发现这两个公式是线性代数方程组。因此可用矩阵乘法表达。例如，输入变量是$$[a_1，a_2，a_3]^T$$ 代表由$$a_1$$，$$a_2$$，$$a_3$$组成的列向量），用向量_a_表示。方程左边是$$[z_1，z_2]^T$$，用向量_z_表示。系数是矩阵_W_。于是，输出公式改写成：

$$
g(W * a) = z
$$

**这个公式就是神经网络从前一层计算后一层的矩阵运算。**

<br></br>



## 两层神经网络 Multi-Layer Perceptron
----
两层神经网络除了包含输入层，输出层外，还增加了中间层。中间层和输出层都是计算层。

> 其中，$$a_x^{(y)}$$代表第_y_层的第_x_个节点。

<p align="center">
  <img src="./Images/ann7.jpg" width = "400"/>
</p>

<center><i>两层神经网络（中间层计算）</i></center>

<br>

计算最终输出_z_是用中间层$$a_1^{(2)}$$，$$a_2^{(2)}$$和第二个权值矩阵计算得到的：

<p align="center">
  <img src="./Images/ann8.jpg" width = "400"/>
</p>

<center><i>两层神经网络（输出层计算）</i></center>

<br>


### 偏置节点 Bias Unit
偏置节点默认存在，本质上是一个只含有存储功能，且存储值永远为1的单元。在神经网络的每个层次中，除了输出层外，都含有一个偏置单元。

<p align="center">
  <img src="./Images/ann9.jpg" width = "200"/>
</p>

<center><i>两层神经网络（考虑偏置节点）</i></center>

<br>

有些神经网络结构图把偏置节点画出，有些不会。考虑偏置节点后的神经网络矩阵运算：
$$
g(W^{(1)} * a^{(1)} + b^{(1)}) = a^{(2)} 
g(W^{(2)} * a^{(2)} + b^{(2)}) = z
$$

**A bias value allows you to shift the activation function to the left or right.**

<p align="center">
  <img src="./Images/ann10.gif" width = "200"/>
</p>

<center><i>1-input, 1-output network with no bias</i></center>

<br>

The output is computed by multiplying input _x_ by weight $$w_0$$ and passing result through activation function:

<p align="center">
  <img src="./Images/ann11.png" width = "400"/>
</p>

Changing weight $$w_0$$ essentially changes the "steepness" of the sigmoid. That's useful, but what if you wanted to output 0 when _x_ is 2? Just changing steepness of sigmoid won't really work -- you want to be able to shift entire curve to right.

That's exactly what bias to do. If we add a bias, like so:

<p align="center">
  <img src="./Images/ann12.gif" width = "200"/>
</p>

<br>

Output becomes $$sig(w_0*x + w_1*1.0)$$. Here is what the output looks like for various values of $$w_1$$:

<p align="center">
  <img src="./Images/ann13.png" width = "400"/>
</p>

<br>


### 效果
与单层神经网络不同，理论证明，两层神经网络可无限逼近任意连续函数。即，面对复杂非线性分类任务，两层神经网络分类的很好。

<p align="center">
  <img src="./Images/ann14.png" width = "400"/>
</p>

<center><i>两层神经网络（决策分界）</i></center>

<br>

两层神经网络决策分界是平滑的曲线。单层网络只能做线性分类任务，而两层网络中后一层也是线性分类层。为什么两个线性分类结合可做非线性分类？

把输出层决策分界单独拿出来看：

<p align="center">
  <img src="./Images/ann15.png" width = "400"/>
</p>

<center><i>两层神经网络（空间变换）</i></center>

<br>

输出层决策分界仍是直线。关键是从输入层到隐藏层时，数据发生空间变换。即，隐藏层对原始数据空间变换，使其可被线性分类。然后输出层决策分界划出一个线性分类分界线，对其进行分类。联想到推导出的矩阵公式，矩阵和向量相乘，本质上是对向量坐标空间进行变换。**因此，隐藏层参数矩阵使数据原始坐标空间从线性不可分转成线性可分。**

**两层神经网络通过两层线性模型模拟数据内真实的非线性函数。因此，多层神经网络本质是复杂函数拟合。**

<br></br>



## 多层神经网络 Deep Learning
----
### 普通多层神经网络结构
可看出$$W^{(1)}$$有6个参数，$$W^{(2)}$$有4个参数，$$ W^{(3)}$$有6个参数，所以整个神经网络中参数有16个（不考虑偏置节点）。

<p align="center">
  <img src="./Images/ann16.jpg" width = "400"/>
</p>

<center><i>多层神经网络（较少参数）</i></center>

<br>

将中间层节点数做一下调整。第一个中间层改为3个单元，第二个中间层改为4个单元。调整后，整个网络参数变成33个。

<p align="center">
  <img src="./Images/ann17.jpg" width = "400"/>
</p>

<center><i>多层神经网络（较多参数）</i></center>

<br>

虽然层数不变，但第二个神经网络参数数量是第一个两倍，从而带来了更好的表示能力。在参数一致的情况下，可以获得一个“更深”的网络：

<p align="center">
  <img src="./Images/ann18.jpg" width = "400"/>
</p>

<center><i>多层神经网络（更深的层次）</i></center>

<br>

虽然参数仍是33，但有4个中间层。意味着一样的参数数量，可用更深的层次去表达。

<br>


### 效果
增加更多层次的好处是：
1. 更深入的表示特征

    网络层数增加，每一层对前一层抽象表示更深入。在神经网络中，每一层神经元学习前一层神经元值的更抽象表示。例如，第一个隐藏层学习“边缘”特征，第二个隐藏层学习由“边缘”组成的“形状”特征，第三个隐藏层学习由“形状”组成的“图案”的特征，最后的隐藏层学习由“图案”组成的“目标”特征。

2. 更强的函数模拟能力。

    由于层数增加，整个网络参数就多。神经网络本质是模拟特征与目标间真实关系函数的方法，更多参数意味着模拟的函数可更加复杂，可有更多容量拟合真正关系。

<br>


### 训练
* 单层神经网络使用的激活函数是sgn函数。
* 两层神经网络使用sigmoid函数。
* ReLU函数在训练多层神经网络时，更容易收敛，且预测性能更好。因此，目前深度学习最流行的非线性函数是ReLU函数。

ReLU函数不是传统非线性函数，而是分段线性函数。其表达式$$y=max(x,0)$$在$$x > 0$$，输出是输入；$$x < 0$$，输出0。

深度学习中，泛化比以往更加重要。因为神经网络层数增加，参数也增加，表示能力增强，易出现过拟合。

