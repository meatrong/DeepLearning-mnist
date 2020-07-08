import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print(tf.version)


# ----------------- softmax回归模型（非CNN）  -------------- #
# 定义变量 x,W,b
# x是占位符，2维浮点数[none,784]
x = tf.placeholder("float",[None, 784])
# variable代表可修改的张量：可以计算输入值，也可以在计算中被修改；通常模型参数用Variable标识
# 全为0的张量初始化W和b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# 模型实现 y = Wx + b
# tf.matmul(X，W)表示 x*W
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 实际分布 y_
y_ = tf.placeholder("float",[None, 10])
# 计算交叉熵 reduce_sum:叠加
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练：采用梯度下降算法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 以上 模型设置完毕
# 创建变量 创建一个图，然后在session中启动它
# initialize_all_variables() 一次性为所有变量指定初始值
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 获取数据
# mnist:一个轻量级的类，以Numpy数组的形式存储训练、校验和测试数据集
mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)

#循环训练1000次
for i in range(1000):
    # 随机抓取训练中的100个批处理数据点，然后用数据点作为参数替换占位符运行train_step
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

# 评估模型
# tf.argmax:给出tensor对象在某一维上的数据最大值的索引值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 计算正确率
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))