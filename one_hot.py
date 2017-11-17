import tensorflow as tf

"one_hot学习"
indices = [1, 2, 3, 4]
with tf.Session() as sess:
    x = tf.one_hot(indices=indices, depth=5)
    print(sess.run(x))
