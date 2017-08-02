import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.truncated_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
init = tf.global_variables_initializer()

# model
Y = tf.nn.softmax(tf.matmul(X, W) + b)

# placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None,10])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# X of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(10000):
        #load batch of iamges and correct answers
        batch_X, batch_Y = mnist.train.next_batch(100)
        train_data= {X: batch_X, Y_: batch_Y}

        # train
        sess.run(train_step, feed_dict=train_data) # Run

        if i % 100 == 0:
            correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
