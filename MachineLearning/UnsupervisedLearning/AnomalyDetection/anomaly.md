# <center>Anomaly Detection</center>

<br></br>



## Evaluation
----
异常检测算法是非监督学习算法，意味着无法根据结果变量$$y$$值确定数据是否真的异常。所以，需另一种方法检验算法有效性。开发异常检测系统时，从带标记（异常或正常）数据着手，从其中选择一部分正常数据构建训练集，用剩下正常数据和异常数据混合的数据构成交叉检验集和测试集。

<div style="text-align:center">
    <img src="./Images/evaluation1.png" width="400"></img>
</div>

具体评价方法：
1. 根据测试集数据，估计特征平均值和方差并构建函数。
2. 对交叉检验集，尝试使用不同$$\epsilon$$作为阀值，并预测数据是否异常。根据$$F_1$$或查准率与查全率比例来选择$$\epsilon$$。 
3. 选出$$\epsilon$$后，针对测试集进行预测，计算异常检验系统的值，或查准率与查全率之比。

<div style="text-align:center">
    <img src="./Images/evaluation2.png" width="400"></img>
</div>

<br></br>



## 误差分析和特征选择
----
* 函数转换
    异常检测假设特征符合高斯分布，如果不是，异常检测算法也能工作，但最好将数据转成高斯分布。如用对数函数，$$x=log(x+c)$$，$$c$$为非负常数；或$$x=x^c$$，$$c$$为$$[0, 1]$$间的一个分数。

    ![](./Images/log.jpg)

* 误差分析

    ![](./Images/error1.png)

    一些异常数据可能有较高值，因而被认为正常。通常可将一些相关特征组合，获得新特征（异常数据的该特征值异常地大或小）。如检测计算机状况例子，可用CPU负载与网络通信量比例作为新特征。

    ![](./Images/features1.png)

<br></br>



## Anomaly Detection vs Supervised Learning
----
| 异常检测 | 监督学习 |
| ------- | ------ |
| 数据分布均匀，有大量正负数据。 | 数据非常偏斜，异常样本远少于正常样本。 |
| 异常类型不一，很难根据现有异常样本（即正样本）拟合判断异常样本形态。 | 足够多实例，够用于训练算法。 |

常见场景：

| 异常检测 | 监督学习 |
| ------- | ------ |
| 故障检测 | 垃圾邮件检测 |
| 判断零部件是否异常 | 癌症分类 |

<br></br>



## Exercises
----
![](./Images/ex1.png)


![](./Images/ex2.png)


![](./Images/ex3_1.png)
![](./Images/ex3_2.png)


![](./Images/ex4.png)


![](./Images/ex5.png)


![](./Images/ex6.png)


![](./Images/ex7.png)