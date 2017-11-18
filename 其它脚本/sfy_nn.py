import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 　导入数据集
from tensorflow.examples.tutorials.mnist import input_data  # 导入读取/下载数据集函数

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 读取数据集


# 网络结构函数
def layer(inputs, input_size, output_size, active_function=None):
    W = tf.Variable(tf.truncated_normal(shape=[input_size, output_size], stddev=0.5))  # 初始化权值
    b = tf.Variable(tf.truncated_normal(shape=[1, output_size], stddev=0.1))  # 初始化偏差
    z = tf.matmul(inputs, W) + b

    if active_function:
        return active_function(z)
    else:
        return z


# 定义Session

sess = tf.InteractiveSession()

# 定义网络结构

xs = tf.placeholder(tf.float32, [None, 784])

ys = tf.placeholder(tf.float32, [None, 10])

mlayer1 = layer(xs, 784, 10)

prediction = layer(mlayer1, 10, 10, tf.nn.softmax)
# prediction = layer(xs, 784, 10, tf.nn.softmax)

####### 配置训练方法 #######

# 定义代价函数

J = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), 1))

# 设置优化函数

optimizer = tf.train.GradientDescentOptimizer(0.1)

# 设置优化目标

train = optimizer.minimize(J)

sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义准确率

def compute_accuracy(vs, vy):
    global prediction
    y_prediction = sess.run(prediction, feed_dict={xs: vs})  # sess

    correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(vy, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return sess.run(accuracy)  # 方法


# ------------------------ 训练---------------------#

for i in range(1000):

    data = mnist.train.next_batch(50)

    sess.run(train, feed_dict={xs: data[0], ys: data[1]})

    if i % 20 == 0:
        acc = compute_accuracy(data[0], data[1])

        print("训练准确率：", acc)

facc = accuracy.eval(feed_dict={xs: mnist.test.images, ys: mnist.test.labels})

print("最终——》", facc)
