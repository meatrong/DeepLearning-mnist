# -*- coding: utf-8 -*-
"""
@author: Murong
@project: ex1
@file: test.py
@time: 2020/5/22 11:35
@desc:
"""

from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt


def preimage():
    img = Image.open('testimg.png')
    plt.imshow(img)
    plt.show()
    img = img.convert('L')
    tv = list(img.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
'''
sess = tf.InteractiveSession()
result = preimage()
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())
saver.restore(sess, "./model/model.ckpt")  # 使用模型，参数和之前的代码保持一致

prediction = tf.argmax(y_conv, 1)
predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)

print('识别结果:')
print(predint[0])
'''
'''
#with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./model/model.ckpt")  # 使用模型，参数和之前的代码保持一致

    prediction = tf.argmax(y_conv, 1)
    predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)

    print('识别结果:')
    print(predint[0])

'''
"""
报错：Not found: Key Variable/Adam not found in checkpoint
model中的key
tensor_name: Variable_5/Adam_1
tensor_name: Variable_1
tensor_name: Variable_5/Adam
tensor_name: Variable
tensor_name: Variable_2
tensor_name: Variable_4
tensor_name: Variable_2/Adam_1
tensor_name: Variable_2/Adam
tensor_name: Variable_3/Adam_1
tensor_name: Variable_3
tensor_name: Variable_3/Adam
tensor_name: Variable_8/Adam_1
tensor_name: Variable_6
tensor_name: Variable_4/Adam
tensor_name: Variable_8/Adam
tensor_name: Variable_4/Adam_1
tensor_name: Variable_5
tensor_name: Variable_8
tensor_name: Variable_6/Adam_1
tensor_name: Variable_6/Adam
tensor_name: Variable_7
tensor_name: Variable_7/Adam
tensor_name: Variable_7/Adam_1
tensor_name: Variable_9
tensor_name: Variable_9/Adam_1
tensor_name: Variable_9/Adam
tensor_name: beta1_power
tensor_name: beta2_power
"""