# Created on Tue Sep 20 17:31:22 2016
# @author: omid


import tensorflow as tf

# Download/load the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Defining the symbolic variables

# use tf.placeholder for input
x = tf.placeholder(tf.float32, [None, 784])
# use tf.Variable for model parameters 
W = tf.Variable(tf.zeros([784,10])) 
b = tf.Variable(tf.zeros([10]))

# Defining the softmax model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Defining the cross-entropy training
y_ = tf.placeholder(tf.float32, [None, 10]) #correct answers
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize the tf variables
init = tf.initialize_all_variables()

# Lunch the session
sess = tf.Session()
sess.run(init)

# Train, at last!
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    # Print the accuracy every 50th epoch
    if (i % 50 == 0): 
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        print 'epoch %d, train accuracy = %f' %(i, sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

print 'final test accuracy = %f' %(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
