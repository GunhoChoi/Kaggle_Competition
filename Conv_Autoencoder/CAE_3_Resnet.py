import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image as img
import time
from random import shuffle
start_time = time.time()

path = "./images/"
imagesList = listdir(path)

print("imagesList: %d" % len(imagesList))

images = []
for i in imagesList:
    immg=img.open(path+i)
    immg=immg.resize([320,180])
    images.append(np.asarray(immg).astype(np.float32)/255.0)

images=shuffle(images)


path = "./images_test/"
imagesList_test = listdir(path)

print("imagesList_test: %d" % len(imagesList_test))

images_test = []
for i in imagesList_test:
    immg=img.open(path+i)
    immg=immg.resize([320,180])
    images_test.append(np.asarray(immg).astype(np.float32)/255.0)
   
# hyperparameters

batch_size=16
learning_rate = 1e-2
iteration=10000


X = tf.placeholder(dtype=tf.float32, shape=[None, 180, 320, 3])

layer1 = tf.contrib.layers.conv2d(inputs=X, num_outputs=64, kernel_size=7, stride=2, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer1 = tf.contrib.layers.max_pool2d(inputs=layer1, kernel_size=2, stride=2,padding="SAME")

# batch_size,45,80,64

layer2_1 = tf.contrib.layers.conv2d(inputs=layer1, num_outputs=64, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer2_1 = tf.contrib.layers.conv2d(inputs=layer2_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer2_1 = tf.concat([layer1,layer2_1], axis=3)

layer2_2 = tf.contrib.layers.conv2d(inputs=layer2_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer2_2 = tf.contrib.layers.conv2d(inputs=layer2_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer2_2 = tf.concat([layer2_1,layer2_2], axis=3)

layer2_3 = tf.contrib.layers.conv2d(inputs=layer2_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer2_3 = tf.contrib.layers.conv2d(inputs=layer2_3, num_outputs=64, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer2_3 = tf.concat([layer2_2,layer2_3], axis=3) 

# batch_size,45,80,256

layer3_1 = tf.contrib.layers.conv2d(inputs=layer2_3, num_outputs=128, kernel_size=3, stride=2, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer3_1 = tf.contrib.layers.conv2d(inputs=layer3_1, num_outputs=128, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)

layer3_2 = tf.contrib.layers.conv2d(inputs=layer3_1, num_outputs=128, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer3_2 = tf.contrib.layers.conv2d(inputs=layer3_2, num_outputs=128, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer3_2 = tf.concat([layer3_1,layer3_2], axis=3)

layer3_3 = tf.contrib.layers.conv2d(inputs=layer3_2, num_outputs=128, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer3_3 = tf.contrib.layers.conv2d(inputs=layer3_3, num_outputs=128, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer3_3 = tf.concat([layer3_2,layer3_3], axis=3)

layer3_4 = tf.contrib.layers.conv2d(inputs=layer3_3, num_outputs=128, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer3_4 = tf.contrib.layers.conv2d(inputs=layer3_4, num_outputs=128, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer3_4 = tf.concat([layer3_3,layer3_4], axis=3)

# batch_size, 23, 40, 512

layer4_1 = tf.contrib.layers.conv2d(inputs=layer3_4, num_outputs=256, kernel_size=3, stride=2, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_1 = tf.contrib.layers.conv2d(inputs=layer4_1, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)

layer4_2 = tf.contrib.layers.conv2d(inputs=layer4_1, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_2 = tf.contrib.layers.conv2d(inputs=layer4_2, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_2 = tf.concat([layer4_1,layer4_2], axis=3)

layer4_3 = tf.contrib.layers.conv2d(inputs=layer4_2, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_3 = tf.contrib.layers.conv2d(inputs=layer4_3, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_3 = tf.concat([layer4_2,layer4_3], axis=3)

layer4_4 = tf.contrib.layers.conv2d(inputs=layer4_3, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_4 = tf.contrib.layers.conv2d(inputs=layer4_4, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_4 = tf.concat([layer4_3,layer4_4], axis=3)

layer4_5 = tf.contrib.layers.conv2d(inputs=layer4_4, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_5 = tf.contrib.layers.conv2d(inputs=layer4_5, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_5 = tf.concat([layer4_4,layer4_5], axis=3)

layer4_6 = tf.contrib.layers.conv2d(inputs=layer4_5, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_6 = tf.contrib.layers.conv2d(inputs=layer4_6, num_outputs=256, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer4_6 = tf.concat([layer4_5,layer4_6], axis=3)

# batch_size,12,20,1536

layer5_1 = tf.contrib.layers.conv2d(inputs=layer4_6, num_outputs=512, kernel_size=3, stride=2, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer5_1 = tf.contrib.layers.conv2d(inputs=layer5_1, num_outputs=512, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)

layer5_2 = tf.contrib.layers.conv2d(inputs=layer5_1, num_outputs=512, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer5_2 = tf.contrib.layers.conv2d(inputs=layer5_2, num_outputs=512, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer5_2 = tf.concat([layer5_1,layer5_2], axis=3)

layer5_3 = tf.contrib.layers.conv2d(inputs=layer5_2, num_outputs=512, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer5_3 = tf.contrib.layers.conv2d(inputs=layer5_3, num_outputs=512, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu ,normalizer_fn=tf.contrib.layers.batch_norm)
layer5_3 = tf.concat([layer5_2,layer5_3], axis=3)

# batch,6,10,1536

avg_layer = tf.contrib.layers.avg_pool2d(layer5_3, kernel_size=[6,10], stride=1, padding="VALID")
avg_layer = tf.reshape(avg_layer, shape=[-1])

# encoded to 1536 numbers

encoded = avg_layer 


trans_layer1 = tf.reshape(encoded, shape=[-1,1,1,1536])
trans_layer1 = tf.contrib.layers.conv2d(inputs=trans_layer1, num_outputs=6*10*512, kernel_size=1)
trans_layer1 = tf.reshape(trans_layer1, shape=[-1,6,10,512])

trans_layer2_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer1, num_outputs=512, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer2_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer2_1, num_outputs=512, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer2_1 = tf.concat([trans_layer1,trans_layer2_1],axis=3)

trans_layer2_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer2_1, num_outputs=512, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer2_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer2_2, num_outputs=512, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer2_2 = tf.concat([trans_layer2_1,trans_layer2_2],axis=3)

trans_layer2_3 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer2_2, num_outputs=512, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm, stride=2) # 12,20,256

# batch_size,12,20,512

trans_layer3_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer2_3, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_1, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_1 = tf.concat([trans_layer2_3,trans_layer3_1],axis=3)

trans_layer3_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_1, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_2, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_2 = tf.concat([trans_layer3_1,trans_layer3_2],axis=3)

trans_layer3_3 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_2, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_3 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_3, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_3 = tf.concat([trans_layer3_2,trans_layer3_3],axis=3)

trans_layer3_4 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_3, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_4 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_4, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_4 = tf.concat([trans_layer3_3,trans_layer3_4],axis=3)

trans_layer3_5 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_4, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_5 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_5, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_5 = tf.concat([trans_layer3_4,trans_layer3_5],axis=3)

trans_layer3_6 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_5, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_6 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_6, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer3_6 = tf.concat([trans_layer3_5,trans_layer3_6],axis=3)

trans_layer3_7 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_6, num_outputs=256, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm, stride=2) #24,40,128
trans_layer3_7 = tf.slice(trans_layer3_7, begin=[0,0,0,0],size=[batch_size,23,40,256]) 

# batch_ize, 23, 40, 128

trans_layer4_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer3_7, num_outputs=128, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer4_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_1, num_outputs=128, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer4_1 = tf.concat([trans_layer3_7,trans_layer4_1],axis=3)

trans_layer4_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_1, num_outputs=128, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer4_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_2, num_outputs=128, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer4_2 = tf.concat([trans_layer4_1,trans_layer4_2],axis=3)

trans_layer4_3 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_2, num_outputs=128, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer4_3 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_3, num_outputs=128, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer4_3 = tf.concat([trans_layer4_2,trans_layer4_3],axis=3)

trans_layer4_4 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_3, num_outputs=128, kernel_size=3, normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer4_4 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_4, num_outputs=128, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer4_4 = tf.concat([trans_layer4_3,trans_layer4_4],axis=3)

trans_layer4_5 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_4, num_outputs=128, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm, stride=2) # 46,80,64
trans_layer4_5 = tf.slice(trans_layer4_5, begin=[0,0,0,0],size=[batch_size,45,80,128]) 

# batch_size, 45, 80, 64

trans_layer5_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer4_5, num_outputs=64, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer5_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer5_1, num_outputs=64, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer5_1 = tf.concat([trans_layer4_5,trans_layer5_1],axis=3)

trans_layer5_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer5_1, num_outputs=64, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer5_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer5_2, num_outputs=64, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer5_2 = tf.concat([trans_layer5_1,trans_layer5_2],axis=3)

trans_layer5_3 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer5_2, num_outputs=64, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer5_3 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer5_3, num_outputs=64, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer5_3 = tf.concat([trans_layer5_2,trans_layer5_3],axis=3)

trans_layer5_4 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer5_3, num_outputs=64, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm, stride=2) 

# batch_size, 90,160,32

trans_layer6_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer5_4, num_outputs=3, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer6_1 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer6_1, num_outputs=3, kernel_size=3,normalizer_fn=tf.contrib.layers.batch_norm)
trans_layer6_1 = tf.concat([trans_layer5_4, trans_layer6_1],axis=3)

trans_layer6_2 = tf.contrib.layers.conv2d_transpose(inputs=trans_layer6_1, num_outputs=3, kernel_size=7,normalizer_fn=tf.contrib.layers.batch_norm, stride=2) # 180,320,3

decoder_output = trans_layer6_2

# calculate loss and optimize

loss = tf.reduce_sum(tf.abs(decoder_output-X))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(iteration):
        
        for i in range(int(len(images)/batch_size)):

            image_feed=images[i:i+batch_size]
            _, loss_num_train,enc_result_train, dec_result_train = sess.run([optimizer, loss, encoded, decoder_output], feed_dict={X: image_feed})

            
            if epoch % 10 == 0 and epoch is not 0:
	            
                saver.save(sess, './model/model.ckpt')

                for i in range(int(len(images_test)/batch_size)):

                    image_feed_test=images_test[i:i+batch_size]
                    loss_num,enc_result, dec_result = sess.run([loss, encoded, decoder_output], feed_dict={X: image_feed_test})
                    #print((dec_result[0].shape))
                    plt.imsave("epoch:"+str(epoch)+"_"+imagesList_test[i*batch_size][:-3]+"png",dec_result[0])
		            

        print("{} loss: {}".format(epoch, loss_num_train))
       
    print("--- %s seconds ---" % (time.time() - start_time))
	