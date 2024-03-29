# <center>Supervised Learning</center>

<br></br>



## 生成模型与判别模型 Gernerative vs Discriminative
----
监督学习的任务是学习一个模型，对给定的输入预测相应的输出。模型一般形式为一个决策函数或一个条件概率分布（后验概率）：
* 决策函数：输入$$X$$返回$$Y$$，$$Y$$与阈值比较，判定$$X$$类别。
* 条件概率分布：输入$$X$$返回属于每个类别的概率，将概率最大的作为$$X$$所属类别。

监督学习模型可分为生成模型与判别模型：
* 判别模型直接学习决策函数或条件概率分布。

    判别模型学习的是类之间最优分隔面，反映不同类数据间差异。比如K近邻、神经网络、决策树、逻辑斯蒂回归、最大熵模型和SVM等。

    优点是直接面对预测，学习准确率更高。由于直接学习$$P(Y|X)$$或$$f(X)$$，可对数据进行各种抽象，定义特征并使用特征，以简化学习过程。

    缺点是不能反映训练数据本身的特性。

* 生成模型学习的是联合概率分布$$P(X,Y)$$，根据条件概率计算$$P(Y|X)$$。

    常见的有朴素贝叶斯、隐马尔可夫模型、混合高斯模型和贝叶斯网络等。

    优点是可还原联合概率分布$$P(X,Y)$$，而判别方法不能。另外，学习收敛速度更快，即当样本容量增加时，学到的模型可更快收敛到真实模型。

    缺点是学习和计算过程复杂。

注意，由生成模型可得到判别模型，但反向不行。存在“隐变量”时，只能使用生成模型。

 > 当找不到引起某一现象原因时，就把起作用但无法确定的因素，叫“隐变量”

<br></br>
