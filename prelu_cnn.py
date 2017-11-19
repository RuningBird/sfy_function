import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

MAX_ITER = 2000
MAX_TEST = 2000

### 定义数据流图对象 ###
sess = tf.InteractiveSession()


### 1.1 定义构造函数：卷积核|权值 ###

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


### 1.2 定义构造函数：偏置 ###
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape) + 0.1)


### 1.3 定义卷积函数 ##
def conv2d(inputs, kernel):
    return tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')


### 1.4 定义池化函数
def max_pool_2x2(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


### 1.3 定义placeholder ###
x_holder = tf.placeholder(tf.float32, [None, 784])
y_holder = tf.placeholder(tf.float32, [None, 10])

### 2 定义网络结构 ###

x_image = tf.reshape(x_holder, [-1, 28, 28, 1])  # 重整输出数据

### 2.1 卷积层1 ###
kernel_conv1 = weight_variable([5, 5, 1, 32])  # 32个【5*5 】的卷积核
bias_conv1 = bias_variable([32])

# h_conv1 = tf.nn.relu(conv2d(x_image, kernel_conv1) + bias_conv1) # ReLU
# h_conv1 = conv2d(x_image, kernel_conv1) + bias_conv1 # SReLU
# h_conv1 = tf.nn.sigmoid(conv2d(x_image, kernel_conv1)) + bias_conv1 # Sigmoid
# h_conv1 = tf.nn.tanh(conv2d(x_image, kernel_conv1)) + bias_conv1  # tanh

h_conv1 = 0.1 * tf.nn.relu(conv2d(x_image, kernel_conv1) + bias_conv1)  # PReLU
h_poo1l = max_pool_2x2(h_conv1)
## 此时图像为：32*【14，14】



### 2.2 卷积层2 ###
kernel_conv2 = weight_variable([5, 5, 32, 64])
bias_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_poo1l, kernel_conv2) + bias_conv2)
h_poo12 = max_pool_2x2(h_conv2)
## 此时图像为：64*【7，7】

### 2.3 全连接层 ###
fc1_wight = weight_variable([7 * 7 * 64, 1024])
fc1_bias = bias_variable([1024])
h_poo12_flat = tf.reshape(h_poo12, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_poo12_flat, fc1_wight) + fc1_bias)

### 2.4 dropout ###
drop_prob = tf.placeholder(tf.float32)
h_dropout = tf.nn.dropout(h_fc1, keep_prob=drop_prob)

### 2.5 softmax ###
fc2_weight = weight_variable([1024, 10])
fc2_bias = bias_variable([10])
y_prediction = tf.nn.softmax(tf.matmul(h_dropout, fc2_weight) + fc2_bias)

################################## 4, 学习方法设置 ###############################

### 4.1 定义代价函数：corss | loss
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction)))
cross_entropy = -tf.reduce_sum(y_holder * tf.log(y_prediction))

### 4.2 定义学习（优化）方法 ###
optimizer = tf.train.AdamOptimizer(1e-4)
# optimizer = tf.train.GradientDescentOptimizer(0.1)

## 4.3 定义学习（优化）方向 ###
train_step = optimizer.minimize(cross_entropy)
################################## 5, 评估方法设置#######################################
correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y_holder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
################################## 6, 训练 #######################################

sess.run(tf.global_variables_initializer())  # 初始化全部变量

# 训练过程

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 读取数据集

x_data = []  # 绘图横坐标
arr_train_cross = []  # 代价函数数组
arr_train_acc = []  # 训练准确率
arr_test_acc = []  # 测试准确率

for i in range(MAX_ITER):
    batch = mnist.train.next_batch(50)
    test_batch = mnist.test.next_batch(20)

    # print('\n', sess.run(y_prediction, feed_dict={x_holder: batch[0], y_holder: batch[1], drop_prob: 1}))
    # print('==============================================')

    train_step.run(feed_dict={x_holder: batch[0], y_holder: batch[1], drop_prob: 0.5})

    if i % 20 == 0:
        x_data.append(i)
        # 计算训练准确率
        step_sfyin_acc = accuracy.eval(feed_dict={x_holder: batch[0], y_holder: batch[1], drop_prob: 1.0})
        arr_train_acc.append(step_sfyin_acc)

        # 计算测试准确率
        step_test_acc = accuracy.eval(feed_dict={x_holder: test_batch[0], y_holder: test_batch[1], drop_prob: 1.0})
        arr_test_acc.append(step_test_acc)

        # 计算代价函数
        temp_j = cross_entropy.eval(feed_dict={x_holder: batch[0], y_holder: batch[1], drop_prob: 1.0})
        arr_train_cross.append(temp_j)
        # # print("代价：", temp_j)

    if i % 50 == 0:
        print("at step %s ,accuracy is %s" % (i, step_sfyin_acc))

test_batch = mnist.test.next_batch(MAX_TEST)
print("最终准确率为： %s", accuracy.eval(feed_dict={x_holder: test_batch[0], y_holder: test_batch[1], drop_prob: 1.0}))

# 写入文件

# 读取当前文件运行次数并命名
num = 0
train_name = 'prelu'
n_path = '/简化激活函数/datas/'
save_path = n_path + train_name + '_cnn/'
with open(n_path + train_name + "_n.txt", 'r') as f:
    ns = f.readline()
    num = int(ns)
    print(num)

# 1，保存代价函数数值
# np_x_data = np.array(x_data)

np_sfy_j_y = np.array(arr_train_cross)
np.savetxt(save_path + train_name + "_j_" + str(num) + ".csv", [np_sfy_j_y], delimiter=',')

# 2,保存训练过程中的acc
np_sfy_train_acc = np.array(arr_train_acc)
np_sfy_test_acc = np.array(arr_test_acc)
np.savetxt(save_path + train_name + "_train_acc_" + str(num) + ".csv", [np_sfy_train_acc, np_sfy_test_acc],
           delimiter=',')

# 更新运行次数，放在最后防止提前写入运行失败
with open(n_path + train_name + "_n.txt", 'w') as f:
    tmp = num + 1
    f.write(str(tmp))

# 绘图
# fig = plt.figure(1)
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(x_data, sfy_test_acc)
plt.title('PReLU function ACC performance')
plt.plot(x_data, arr_train_acc, x_data, arr_test_acc)
plt.legend(['train_acc', 'test_acc'])
plt.savefig(save_path + "train_vs_test_acc_" + str(num) + ".png")

plt.show()
