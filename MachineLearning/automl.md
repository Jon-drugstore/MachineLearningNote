# <center>AutoML</center>

<br></br>



## Feature Engineering
----  
### feature-tools  
https://github.com/Featuretools/featuretools

用关系数据库数据集中的模式解决特征工程。使用深度特征合成（DFS）算法，遍历关系数据库模式描述的数据关系路径。DFS遍历时，通过应用于数据的操作（如sum、average、count）生成合成特征。

例如，它可对来自给定客户ID的事务列表进行sum操作。不过这是一个深度操作，算法可遍历更深层特征。featuretools优势在于可靠性及在使用时间序列数据时处理信息泄漏的能力。

<br>


### boruta-py
https://github.com/scikit-learn-contrib/boruta_py

是Brouta特征消减策略的实现。其中问题以“完全相关”方式构建，算法保留对模型有显著贡献的所有特征。与其他很多特征消减算法使用的“最小化最优”特征集相反。

Boruta通过创建由目标特征的随机排序值组成的合成特征来确定特征重要性，然后在原始特征集上训练基于简单树的分类器和在合成特征中替换目标特征的特征集。所有特征的性能差异被用于计算相关性。

<br>


### categorical-encoding
https://github.com/scikit-learn-contrib/categorical-encoding

这个库扩展很多实现scikit-learn数据转换器接口的分类编码方法，并实现常见分类编码方法。还可直接与pandas一起使用，计算缺失值，及处理训练集之外的变换值。

<br>


### tsfresh
https://github.com/blue-yonder/tsfresh

专注于基于时间序列数据生成特征。它从时间序列数据中提取描述时间序列趋势的特征列表。这些特征包括像方差一样简单的特征和与近似熵一样复杂的特征。这个库能从数据中提取趋势，让机器学习算法更易解释时间序列数据集。它通过假设检验获取大量生成的特征集，并将其消减到最能解释趋势的特征。

tsfresh与pandas和sklearn兼容，主要功能是它的可伸缩数据处理能力。

<br>


### Trane
https://github.com/HDI-Project/Trane

可处理存储在关系数据库中的时间序列数据，用于表述时间序列问题。数据科学家可通过指定数据集元信息让这个引擎表述数据库的时间序列数据的监督问题。这个过程通过json描述，数据科学家在文件中描述列和数据类型。框架会处理文件并生成可能的预测问题，而这些问题又用于修改数据集。这个项目可以半自动化的方式生成其他特征。

<br>


### FeatureHub
https://github.com/HDI-Project/FeatureHub

建立在JupyterHub上，可让数据科学家在开发特征工程方法时协作。FeatureHub自动给生成的特征打分，以确定模型总体价值。

<br></br>



## Hyperprarmeter Tuning
----
### Skopt
https://scikit-optimize.github.io/

包括随机搜索、贝叶斯搜索、决策森林和梯度提升树。提供了可靠的优化方法，不过这些模型在给定较小的搜索空间和良好的初始估计值时表现最佳。

<br>


### Hyperopt
https://github.com/hyperopt/hyperopt-sklearn

可调整笨拙的条件或受约束的搜索空间。支持跨机器并行化，用MongoDb作为存储超参数组合结果的中心数据库。这个库通过hyperopt-sklearn和hyperas实现，而这两个模型选择和优化库又分别基于scikit-learn和keras构建。

<br>


### simple(x)
https://github.com/chrisstroemel/Simple  

是贝叶斯优化算法替代方案。与贝叶斯搜索一样，simple(x) 用尽可能少样本进行优化，并将计算复杂度从$$n^{3}$$降低到$$\log{n}$$。因此对大型搜索空间有用。这个库使用单形（$$n$$维三角形）而不是超立方体（$$n$$维立方体）对搜索空间建模，避免计算成本高昂的高斯过程。

<br>


### Ray.tune
https://github.com/ray-project/ray/tree/master/python/ray/tune

主要针对深度学习和强化学习模型。它结合尖端算法，如超频（最低限度训练可用于确定超参数效果的模型算法）、基于人口的训练（在共享超参数同时调整多个模型算法）、响应面算法和中值停止规则（如果模型性能低于中值就停止）。

<br>


### Chocolate
https://github.com/AIworx-Labs/chocolate

是分散的（支持没有中央主节点的并行计算集群）超参数优化库。使用公共数据库联合各个任务的执行，支持网格搜索、随机搜索、准随机搜索、贝叶斯搜索和协方差矩阵自适应进化策略。功能包括支持受约束的搜索空间和优化多个损失函数（多目标优化）。

<br>


### GpFlowOpt
https://github.com/GPflow/GPflowOpt

基于GpFlow的高斯过程优化器。GpFlow使用Tensorflow在GPU运行高斯过程任务的库。

<br>


### FAR-HO
https://github.com/lucfra/FAR-HO

包含一组运行在Tensorflow上的基于梯度的优化器。提供对Tensorflow中基于梯度的超参数优化器访问，允许在GPU或其他计算环境中模型训练和超参数优化。

<br>


### Xcessiv
https://github.com/reiinakano/xcessiv

用于大规模模型开发、执行和集成的框架。能通过单个GUI管理大量机器学习训练、执行和评估。还提供多个集成工具，用于组合模型实现最佳性能。它提供贝叶斯搜索参数优化器，支持高水平并行，且支持与TPOT集成。

<br>


### HORD 
https://github.com/ilija139/HORD

它为黑盒模型生成一个代理函数，并使用代理函数生成可能接近理想的超参数，以消减对完整模型的评估。与parzen estimator、SMAC和高斯过程相比，它表现出更高一致性和更低错误率。适用于极高维度情况。

<br>


### ENAS-pytorch
https://github.com/ilija139/HORD

使用pytorch实现高效的神经架构搜索。通过共享参数实现最快的网络，适用于深度学习架构搜索。

<br></br>



## Full Pipeline
----
### ATM
https://github.com/HDI-Project/ATM

Auto-Tune Models工作量很小。使用穷举搜索和超参数优化执行模型选择。仅支持分类问题，并支持AWS分布式计算。

<br>


### MLBox
https://github.com/AxeldeRomblay/MLBox

除现有框架已实现的特征工程外，还提供数据收集、数据清理和训练测试漂移检测。使用Tree Parzen Estimator优化模型类型超参数。

<br>

### auto_ml
https://github.com/ClimbsRocks/auto_ml

使用基于进化网格搜索方法完成特征处理和模型优化。通过利用高度优化库，如XGBoost、TensorFlow、Keras、LightGBM和sklearn，提高速度。

<br>

### auto-sklearn
https://github.com/automl/auto-sklearn

使用贝叶斯搜索优化机器学习管道中使用的数据预处理器、特征预处理器和分类器。多个管道经过训练并整合成一个完整模型。

<br>


### H2O
https://github.com/h2oai/h2o-3

Java开发的机器学习平台，在与机器学习库类似的抽象级别上运行。提供一个自动机器学习模块，利用自身包含的算法创建机器学习模型。

优势在于能形成大型计算机集群，从而进行大规模伸缩。还支持与Python、JavaScript、Tableau、R和TensorFlow集成。

<br>


### TPOT
https://github.com/EpistasisLab/tpot

TPOT（基于树的管道优化工具）用于查找和生成最佳数据科学管道代码的遗传编程框架。TPOT从sklearn获取算法。TPOT优势在于其独特的优化方法，提供更多独特管道。还提供一个将训练好的管道转换为代码的工具，对于希望进一步调整生成模型的数据科学家是一个好处。
