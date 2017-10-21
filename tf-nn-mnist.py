import tensorflow as tf

#load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/', one_hot=True)

#hyperparameters
learning_rate = 0.001
#hidden layer sizes
h1_size = 600
h2_size = 500
h3_size = 300

n_classes = 10

batch_size = 128
n_epoch = 10

#input
x = tf.placeholder(tf.float32, [None, 784])
#output
y = tf.placeholder(tf.float32, [None, n_classes])

def feed_forward(data):
    h1_layer = {
        'weights': tf.Variable(tf.random_normal([784, h1_size])),
        'biases': tf.Variable(tf.random_normal([h1_size]))
    }
    h2_layer = {
        'weights': tf.Variable(tf.random_normal([h1_size, h2_size])),
        'biases': tf.Variable(tf.random_normal([h2_size]))
    }
    h3_layer = {
        'weights': tf.Variable(tf.random_normal([h2_size, h3_size])),
        'biases': tf.Variable(tf.random_normal([h3_size]))
    }
    output_layer = {
        'weights': tf.Variable(tf.random_normal([h3_size, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    l1 = tf.add(tf.matmul(data, h1_layer['weights']), h1_layer['biases'])
    l1 = tf.nn.relu(l1)


    l2 = tf.add(tf.matmul(l1, h2_layer['weights']), h2_layer['biases'])
    l2 = tf.nn.relu(l2)


    l3 = tf.add(tf.matmul(l2, h3_layer['weights']), h3_layer['biases'])
    l3 = tf.nn.relu(l3)


    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output


def train_nn():
    prediction = feed_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(n_epoch):
            epoch_loss = 0.0
            for _ in range(mnist.train.num_examples // batch_size):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, loss = sess.run([optimizer, cost], {x: epoch_x, y: epoch_y})
                epoch_loss += loss
            print("Epoch", epoch, "of ", n_epoch, "epoches", "loss is: ", epoch_loss)

        #?????????????????!!!!!!!!!!!!!!!!
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print("Accuracy is: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_nn()