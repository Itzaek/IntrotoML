import tensorflow as tf

#Create constants
a = tf.constant(2)
b = tf.constant(3)

#Launch the default graph
with tf.Session() as sess:
    print("a: %i" % sess.run(a), "b: %i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))

#Basic operations by graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

#Define some basic operations
add = tf.add(a,b)
multiply = tf.multiply(a,b)

#Launch the default graphs
with tf.Session() as sess:
    print("Addition with variables: %i" % sess.run(add, feed_dict={a:2,b:3}))
    print("Multiplication with variables: %i " % sess.run(multiply, feed_dict={a:2,b:3}))

#Matrix operations
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
