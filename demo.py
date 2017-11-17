import tensorflow as tf
sess = tf.InteractiveSession()
a = tf.constant(5)
b = tf.constant(5)
c = tf.add(a,b)
d = sess.run(c)
print(d)