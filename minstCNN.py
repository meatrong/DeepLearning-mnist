import os
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import config
from test import preimage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略CPU告警信息
'''
 权重初始化
'''


# 定义方差为0.1的随机数作为卷积核的初始化值（传入参数为卷积核大小）
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义初始值为0.1的片值向量（参数为该向量大小）
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


'''
 卷积层和池化层初始化
'''


# 卷积层:1步长,0边距，strides均为1表示在运算过程中原始的矩阵点一个都不会漏掉
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层: 2*2 最大池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 获取数据，标签变为one_hot形式
mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
train_nums = mnist.train.num_examples
validation_nums = mnist.validation.num_examples
test_nums = mnist.test.num_examples
print('MNIST数据集的个数：')
print(' train_nums = %d' % train_nums, '\n',
      ' validation_nums = %d' % validation_nums, '\n',
      ' test_nums = %d' % test_nums, '\n')

# 构建图， 定义占位符->先分配空间，之后用feed函数填充
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, config.INPUT_NODE])  # None表示大小不定
W = tf.Variable(tf.zeros([config.INPUT_NODE, config.OUTPUT_NODE]))
b = tf.Variable(tf.zeros([config.OUTPUT_NODE]))
# 输出类别值y_也是一个2维张量，
# 其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别
y_ = tf.placeholder(tf.float32, [None, config.OUTPUT_NODE])
# x: 4d, 2,3维:宽、高; 4维:颜色通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1相当于None

result = preimage()

'''
 第一层卷积
'''
# 在5*5的patch中算出32个特征，1个输入通道，32个输出通道各自对应一个偏置量
# 边界处理方式为"SAME"：输入输出相同尺寸
'''
#with tf.name_scope('cov_1') as scope:
#    W_conv1 = weight_variable([5, 5, 1, 32])
#    b_conv1 = bias_variable([32])
#    # Wx+b,再用ReLU激活，最后max pooling
 #   # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#    h_conv1 = conv2d(x_image, W_conv1)
 #   h_relu1 = tf.nn.relu(h_conv1 + b_conv1)
#h_pool1 = max_pool_2x2(h_relu1)
'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
'''
 第二层卷积
'''
# 堆叠类似的层，每个5*5的patch有64个特征

 #   with tf.name_scope('cov_2') as scope:
 #       W_conv2 = weight_variable([5, 5, 32, 64])
 #       b_conv2 = bias_variable([64])
 #       # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
 #       h_conv2 = conv2d(h_pool1, W_conv2)
 #       h_relu2 = tf.nn.relu(h_conv2 + b_conv2)
 #   h_pool2 = max_pool_2x2(h_relu2)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
'''
 密集连接层
'''
# 全连接层：减小到7*7，1024个神经元的全连接层,再*w+b 用ReLU激活
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
 Dropout
'''
# Dropout 减少过拟合：用占位符代表一个神经元的输出保持不变的概率
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
 输出层
'''
# 输出层 一个softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 输出预测y_conv预测类别
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

''' 
 训练和评估
'''
# 交叉熵，损失函数是目标类别与预测类别之间的交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 使用ADAM优化器做梯度下降, 学习率1e-4，采用梯度下降最小化交叉熵
train_step = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(cross_entropy)
# 最大值1所在的索引位置就是类别标签，
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(y_, 1))
# tf.argmax(input, axis)根据axis取值的不同返回每行或者每列最大值的索引,
# axis = 0时，返回每列的最大值的索引，axis=1时，返回每行最大值的索引，
# 若数组长度不一致，axis最大值为最小数组的长度-1；不一致时，axis=0即为找到数组之和最大值的索引

# 将布尔值转换为浮点数，然后取平均值得出分类准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

'''画图'''
# 显示loss和accuracy的变化趋势
tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
# 显示全连接层的权重和第一层卷积核的统计直方图
tf.summary.histogram("weight", W_fc2)
tf.summary.histogram("nn", W_conv1)
# 显示训练的图片
tf.summary.image("train", x_image)

sess.run(tf.initialize_all_variables())

# 对于summary需要先合并，然后把他写入对应的文件夹中，合并后的summary和会话一样也是需要执行的。
merged_summary_op = tf.summary.merge_all()
summary_write = tf.summary.FileWriter("logs/", sess.graph)

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    for i in range(2000):
#        batch = mnist.train.next_batch(50)
#        if i % 100 == 0:
#            train_accuracy = accuracy.eval(feed_dict={
#                x: batch[0], y_: batch[1], keep_prob: 1.0})
#            print('step %d, training accuracy %g' % (i, train_accuracy))
#        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#        summary_str = merged_summary_op.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#        # 将每次执行的summary添加到文件夹中
#        summary_write.add_summary(summary_str, i)
#        summary_write.flush()
#    saver.save(sess, 'model/model.ckpt') #模型储存位置

#    batch = mnist.test.next_batch(50)
#    accuracy_score = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#    print("test accuracy %g" % accuracy_score)

#    prediction = tf.argmax(y_conv, 1)
#    predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0})
#    print('识别结果:')
#    print(predint[0])

for i in range(config.TRAINING_STEPS):
    batch = mnist.train.next_batch(50)  # 每一步迭代加载50个样本
    # 每100次迭代输出一次日志
    if i % 100 == 0:
        train_accuracy = accuracy.eval(  # eval相当于.run()
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        summary_str = merged_summary_op.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # 将每次执行的summary添加到文件夹中
        summary_write.add_summary(summary_str, i)
        summary_write.flush()
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # 开始训练
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# 保存模型
saver.save(sess, 'model/model.ckpt')

# 计算测试数据上的准确率
# 由于测试集过大，会内存溢出导致不能测试，改为分批次测试
batch = mnist.test.next_batch(50)
accuracy_score = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g" % accuracy_score)

prediction = tf.argmax(y_conv, 1)
predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0},session=sess)
print('识别结果:')
print(predint[0])
'''
print("test accuracy %g" %accuracy.eval(
    feed_dict={x:mnist.test.images,
               y_:mnist.test.labels,
               keep_prob:1.0}))
'''

"""
画图
fig = plt.figure(figsize=(12, 9))  # 画布大小为1200*900，默认是800*600，也就是figsize=(8,6)

plt.subplot(211)
plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.show()
plt.grid(True)
#plt.clf()   # clear figure
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.subplot(212)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
fig.tight_layout()
plt.show()
"""
