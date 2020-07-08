# -*- coding: utf-8 -*-
"""
@author: Murong
@project: ex1
@file: kaoshio.py
@time: 2020/6/15 21:09
@desc:
"""
from sklearn.metrics import log_loss
from math import log # 自然对数为底

# 二分类的交叉熵损失函数
# 利用sklearn模块的计算结果
y_true = [0, 1,0, 0, 1]
y_pred = [.2, .8, .4, .1,.9]
sk_log_loss = log_loss(y_true, y_pred)
print('Loss by sklearn: %s.'%sk_log_loss)

# 利用公式计算得到的结果
Loss = 0
for label, prob in zip(y_true, y_pred):
    Loss -= (label*log(prob[0])+(1-label)*log(prob[1]))

Loss = Loss/len(y_true)
print('Loss by equation: %s.'% Loss)
