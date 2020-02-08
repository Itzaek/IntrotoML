import tensorflow as tf

#Create a constant op, the op is added as a node to the default graph
hello = tf.constant('Hello, Tensorflow!')

#Start tf session
session = tf.Session()

#Run graph
print(session.run(hello))
