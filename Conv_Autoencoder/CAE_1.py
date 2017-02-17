import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image
import time
start_time = time.time()

path = "./images/"
imagesList = listdir(path)

print(imagesList)

images = []
for i in imagesList:
    images.append(np.asarray(Image.open(path+i)).astype(np.float32))

#print(images)

X = tf.placeholder(dtype=tf.float32, shape=[None, 480, 640, 3])

# Convolution area

layer1 = tf.contrib.layers.convolution2d(inputs=X, num_outputs=32, kernel_size=3, stride=1, padding="SAME")
layer1_batch = tf.contrib.layers.batch_norm(layer1)
layer1_pool = tf.contrib.layers.max_pool2d(layer1_batch,kernel_size=[2, 2],stride=[2, 2])

layer2 = tf.contrib.layers.convolution2d(inputs=layer1_pool, num_outputs=64, kernel_size=3, stride=1, padding="SAME")
layer2_batch = tf.contrib.layers.batch_norm(layer2)
layer2_pool = tf.contrib.layers.max_pool2d(layer2_batch, kernel_size=[2, 2],stride=[2, 2])

layer3 = tf.contrib.layers.convolution2d(inputs=layer2_pool, num_outputs=128, kernel_size=3, stride=1, padding="SAME")
layer3_batch = tf.contrib.layers.batch_norm(layer3)
layer3_pool = tf.contrib.layers.max_pool2d(layer3_batch, kernel_size=[2, 2],stride=[2, 2])

layer4 = tf.contrib.layers.convolution2d(inputs=layer3_pool, num_outputs=256, kernel_size=3, stride=1, padding="SAME")
layer4_batch = tf.contrib.layers.batch_norm(layer4)
layer4_pool = tf.contrib.layers.max_pool2d(layer4_batch, kernel_size=[2, 2],stride=[2, 2])

layer5 = tf.contrib.layers.convolution2d(inputs=layer4_pool, num_outputs=256, kernel_size=3, stride=1, padding="SAME")
layer5_batch = tf.contrib.layers.batch_norm(layer5)
layer5_pool = tf.contrib.layers.max_pool2d(layer5_batch, kernel_size=[2, 2],stride=[2, 2])

layer6 = tf.contrib.layers.convolution2d(inputs=layer5_pool, num_outputs=128, kernel_size=1, stride=1, padding="SAME")
layer6_batch = tf.contrib.layers.batch_norm(layer6)

# Reshape & Fully connected area

layer6_reshape = tf.reshape(layer6_batch, shape=[-1, 15*20*128])

fc_layer6 = tf.contrib.layers.fully_connected(layer6_reshape, num_outputs=5000, activation_fn=tf.nn.relu)

fc_layer7 = tf.contrib.layers.fully_connected(fc_layer6,num_outputs=100)

output = fc_layer7

fc_layer8 = tf.contrib.layers.fully_connected(output, num_outputs=5000, activation_fn=tf.nn.relu)

fc_layer9 = tf.contrib.layers.fully_connected(fc_layer8, num_outputs=15*20*128, activation_fn=tf.nn.relu)

fc_layer9_reshape = tf.reshape(fc_layer9, shape=[-1, 15, 20, 128])

# Convolution Transpose area

trans_layer1 = tf.contrib.layers.convolution2d_transpose(inputs=fc_layer9_reshape, num_outputs=256, kernel_size=1,stride=1, padding="SAME")
trans_layer1 = tf.contrib.layers.batch_norm(trans_layer1)

trans_layer2 = tf.contrib.layers.convolution2d_transpose(inputs=trans_layer1,  num_outputs=256, kernel_size=2, stride=2, padding="SAME")
trans_layer2 = tf.contrib.layers.batch_norm(trans_layer2)

trans_layer3 = tf.contrib.layers.convolution2d_transpose(inputs=trans_layer2,  num_outputs=128, kernel_size=2, stride=2, padding="SAME")
trans_layer3 = tf.contrib.layers.batch_norm(trans_layer3)

trans_layer4 = tf.contrib.layers.convolution2d_transpose(inputs=trans_layer3,  num_outputs=64, kernel_size=2, stride=2, padding="SAME")
trans_layer4 = tf.contrib.layers.batch_norm(trans_layer4)

trans_layer5 = tf.contrib.layers.convolution2d_transpose(inputs=trans_layer4,  num_outputs=32, kernel_size=2, stride=2, padding="SAME")
trans_layer5 = tf.contrib.layers.batch_norm(trans_layer5)

trans_layer6 = tf.contrib.layers.convolution2d_transpose(inputs=trans_layer5,  num_outputs=1, kernel_size=2, stride=2, padding="SAME")

decoder_output = trans_layer6

loss = tf.reduce_sum(tf.square(decoder_output-X))

learning_rate = 1e-2

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(10000):
        _, loss_num = sess.run([optimizer, loss], feed_dict={X: images})

        if epoch % 100 == 0:
            print("{} loss: {}".format(epoch, loss_num))

    print("--- %s seconds ---" % (time.time() - start_time))

    a = sess.run(trans_layer6, feed_dict={X: images})

    list_layers = [layer1_batch, layer2_batch, layer3_batch, layer4_batch, layer5_batch]

    for i in list_layers:
        b = sess.run(i, feed_dict={X: images})

        for j in range(3):
            check = b[0, :, :, j]
            plt.imshow(check, cmap='gray')
            plt.show()

    plt.imshow(a[0, :, :, 0])
    plt.show()
    plt.imsave("CAE_test1.png", a[0, :, :, 0])
