import tensorflow as tf
import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x,[-1,28,28,1])

def weight_tensor(shape):
    init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)
def bias_tensor(shape):
    init = tf.constant(0.1,shape = shape)
    return tf.Variable(init)

W_conv1 = weight_tensor([5,5,1,32])
b_conv1 = bias_tensor([32])
a_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
a_pool1 = tf.nn.max_pool(a_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_conv2 = weight_tensor([5,5,32,64])
b_conv2 = bias_tensor([64])
a_conv2 = tf.nn.relu(tf.nn.conv2d(a_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
a_pool2 = tf.nn.max_pool(a_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_fc1 = weight_tensor([7*7*64,1024])
b_fc1 = bias_tensor([1024])
a_flat = tf.reshape(a_pool2,[-1,7*7*64])
a_fc1 = tf.nn.relu(tf.matmul(a_flat,W_fc1)+b_fc1)
a_drop = tf.nn.dropout(a_fc1,keep_prob)

W_fc2 = weight_tensor([1024,10])
b_fc2 = bias_tensor([10])
y_pred = tf.nn.softmax(tf.matmul(a_drop,W_fc2)+b_fc2)


train_accuracys = []
cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))
train_step = tf.train.AdagradOptimizer(0.0001).minimize(cross_entropy)
correct_ratio = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_ratio,'float'))

sess = tf.Session()
# summary_op = tf.summary.merge_all()
# summary_writer = tf.summary.FileWriter('MNIST_log', sess.graph)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y:batch[1],keep_prob:1})
        train_accuracys.append(train_accuracy)
        print('step %d,training accuracy %g' % (i, train_accuracy))
        # if i%1000 == 0:
        #     summary_str = sess.run(summary_op)
        #     summary_writer.add_summary(summary_str, i)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})


print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1}))





