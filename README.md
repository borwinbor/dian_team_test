# 学习记录
这个文档用于存储个人的学习历程
## 3/14
速通了python的基本语法,了解了深度学习的概念与用途
## 3/15
学习了python中numpy pandas等库函数的应用，开始学习深度神经网络的完整训练流程
## 3/16
开始学习tensorflow2,并开始依赖其建立深度学习模型
### 实现了全连接神经网络的搭建与评估准确率随迭代次数的变化<br>
[NN.py](https://github.com/borwinbor/dian_team_test/blob/main/NN.py)<br>
准确率随时间变化的图像<br>
<img src="https://github.com/borwinbor/dian_team_test/blob/main/%E5%87%86%E7%A1%AE%E7%8E%87%E9%9A%8F%E8%BF%AD%E4%BB%A3%E6%AC%A1%E6%95%B0%E5%8F%98%E5%8C%96%E5%9B%BE%E5%83%8F.png" width=800>
## 3/17
用库函数函数对每次迭代后的四项指标进行了计算(虽然写了自定义的指标评测函数，但是用自定义函数计算一直出现评测指标在1几乎不变的情况qwq)<br>
<img src="https://github.com/borwinbor/dian_team_test/blob/main/NN.png" width=800>
## 3/19
发现了之前评测指标一直在1不变的原因：没有合理的加权平均方式，之前一直直接对精确度求平均值，在改用根据样本数量进行加权后，成功实现了评测,但是还是只实现了随着迭代层数评测指标的变化<br>
[NN(自定义函数实现版本).ipynb](https://github.com/borwinbor/dian_team_test/blob/main/NN(%E8%87%AA%E5%AE%9A%E4%B9%89%E5%87%BD%E6%95%B0%E5%AE%9E%E7%8E%B0%E7%89%88%E6%9C%AC).ipynb)<br>
赶在ddl之前勉强写出了RNN实现分类并进行评测的代码
