import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# avoid dead neurons by giving slightly positive init value
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape) 
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True)
	#ksize is window size, strides is [batch height width channel]
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
	train_file='test.csv'
	input_data=genfromtxt(train_file, delimiter=',')
	image=input_data[1:,:].astype("float32")

	saved_model=tf.train.import_meta_graph("model/model-1.0.meta")
	saved_model.restore(sess,"model/model-1.0")

	x =tf.placeholder(tf.float32, shape=[None,784],name='x')
	y_=tf.placeholder(tf.float32, shape=[None,10],name="y_")
	x_image = tf.reshape(x, [-1,28,28,1])

	# use variables from loaded model
	W_conv1=tf.Variable(sess.graph.get_tensor_by_name("W_conv1:0"))
	b_conv1=tf.Variable(sess.graph.get_tensor_by_name("b_conv1:0"))
	W_conv2=tf.Variable(sess.graph.get_tensor_by_name("W_conv2:0"))
	b_conv2=tf.Variable(sess.graph.get_tensor_by_name("b_conv2:0"))
	
	W_conv3=tf.Variable(sess.graph.get_tensor_by_name("W_conv3:0"))
	b_conv3=tf.Variable(sess.graph.get_tensor_by_name("b_conv3:0"))
	W_conv4=tf.Variable(sess.graph.get_tensor_by_name("W_conv4:0"))
	b_conv4=tf.Variable(sess.graph.get_tensor_by_name("b_conv4:0"))
		
	W_fc1=tf.Variable(sess.graph.get_tensor_by_name("W_fc1:0"))
	b_fc1=tf.Variable(sess.graph.get_tensor_by_name("b_fc1:0"))
	W_fc2=tf.Variable(sess.graph.get_tensor_by_name("W_fc2:0"))
	b_fc2=tf.Variable(sess.graph.get_tensor_by_name("b_fc2:0"))

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
	h_pool1 = max_pool_2x2(h_conv2)	#28x28 -> 14x14
	h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
	h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
	h_pool2 = max_pool_2x2(h_conv4)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*128]) # 7*7*64 = 3136
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	
	sess.run(tf.global_variables_initializer())

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	#train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
	correct_prediction = tf.argmax(y,1)
	
	batch_size=1

	result_stack=[]

	j=0
	while j<len(image):
		#print(j)
		
		batch_image=image[j:j+batch_size]
		feed_dict={x:batch_image,keep_prob:1.0}
		j+=batch_size
		_,result =sess.run([y,correct_prediction],feed_dict=feed_dict)

		'''a=np.array(batch_image[0])
			a=a.reshape((28,28))
			plt.imshow(a)
			plt.show()'''
		result_stack.append(int(result))

	np.savetxt("result_6layer_1.csv",result_stack,delimiter=",")
