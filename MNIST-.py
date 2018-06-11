import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)



x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,10])
x_image = tf.reshape(x,[-1,28,28,1])

#================conv1+maxpool1==================
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.5))
b_conv1 = tf.Variable(tf.constant(1,'float',[32]))
a_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
a_pool1 = tf.nn.max_pool(a_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#================conv2+maxpool2==================
W_conv2 = tf.Variable(tf.truncated_normal([3,3,32,32],stddev=0.5))
b_conv2 = tf.Variable(tf.constant(1,'float',[32]))
a_conv2 = tf.nn.relu(tf.nn.conv2d(a_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
a_pool2 = tf.nn.max_pool(a_conv2,ksize=[1,3,3,1],strides=[1,1,1,1],padding = 'SAME')
#================conv3+maxpool3==================
W_conv3 = tf.Variable(tf.truncated_normal([3,3,32,64],stddev=0.5))
b_conv3 = tf.Variable(tf.constant(1,'float',[64]))
a_conv3 = tf.nn.relu(tf.nn.conv2d(a_pool2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
a_pool3 = tf.nn.max_pool(a_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
#====================FC1+dropout=================
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024]))
b_fc1 = tf.Variable(tf.constant(1,'float',[1024]))
a_flat = tf.reshape(a_pool3,[-1,7*7*64])
a_fc1 = tf.nn.relu(tf.matmul(a_flat,W_fc1)+b_fc1)
keep_prob = tf.placeholder('float')
a_fc1_drop = tf.nn.dropout(a_fc1,keep_prob)
#====================FC2+dropout=================
W_fc2 = tf.Variable(tf.truncated_normal([1024,10]))
b_fc2 = tf.Variable(tf.constant(1,'float',[10]))
y_pred = tf.nn.softmax(tf.matmul(a_fc1_drop,W_fc2)+b_fc2)



cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))
train_step = tf.train.AdagradOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(3000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y:batch[1],keep_prob:1.0})
        print('step %d,training accuracy %g'%(i,train_accuracy))
    sess.run(train_step,feed_dict={x:batch[0],y:batch[1],keep_prob:0.5})





