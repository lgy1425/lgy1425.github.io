---
layout: post
title:  "Paper Review : Super Resolution"
categories: deep-learning
---

Super resolution is a technique for generating high resolution images from low resolution images. You can develop super resolution with better performance when using deep learning than traditional machine learning or computer vision methods. An article named “Super resolution via deep learning”(<a>https://arxiv.org/abs/1706.09077</a>) introduced summary of super resolution techniques in single image, video. In my article, I detail some deep learning networks for super resolution and implement it.Below table is the list of deep learning and machine learning methods for benchmark.

![](https://i.imgur.com/9uB0XMH.png)

All of experiments were done with ImageNet Data from URLs of Fall 2011 Release at <a>http://image-net.org/download-imageurls</a> . Images are randomly cropped as 200 x 200px and resized 100 x 100px(scale factor 2) or 66 x 66px(scale factor 3) or 50 x 50 px(scale factor 4) and blurred. All of networks made low resolution images become high resolution(200x200px) images. Performance index is PSNR(Peak signal-to-noise ratio) and implemented by skit-image package.

![](https://i.imgur.com/EipcCbj.png)

```python
import os
from PIL import Image,ImageFilter 
import random
from skimage.measure import compare_psnr
import numpy as np

# original loading with PIL package
origin = Image.open("img/1.jpg")

#random crop --> generate 200x200px image
width = origin.size[0]
height = origin.size[1]
x = width * 0.1 + random.randint(0,int(width*0.8-200))
y = height * 0.1 + random.randint(0,int(height*0.8-200))
cropped = origin.crop((x,y,x+200,y+200))
high_resolution = cropped

# generate row resolution image
low_resolution = cropped.resize((100,100)).filter(ImageFilter.BLUR)

# super resolution with bicubic interpolation
super_resolution = low_resolution.resize((200,200),Image.BICUBIC)

# get psnr
high_resolution = np.array(high_resolution)
super_resolution = np.array(super_resolution)
print compare_psnr(high_resolution,super_resolution)
```


<h3>SRCNN(<a>https://arxiv.org/pdf/1501.00092.pdf</a>)</h3>

![](https://i.imgur.com/z4Lgnob.png)

SRCNN(Super Resolution Convolution Network) was the first CNN for super resolution and it was inspired by CNN employed for image classification. It has very simple architecture that has 3 convolution layers. Its implementation was produced by Matlab and Caffe code. I imitated their code with Tensorflow like below. SRCNN’s output is smaller than original image to avoid border effect.

```python

# Implementation of SRCNN

X = tf.placeholder("float", [None, None,None,3],name="x")
Y = tf.placeholder("float", [None, 188,188,3],name="y")

weights = {
    "w_1" : init_weights([9,9,3,64],"w_1"),
    "w_2" : init_weights([1,1,64,32],"w_2"),
    "w_3" : init_weights([5,5,32,3],"w_3")
}
bias = {
    "b_1" : init_weights([64],"b_1"),
    "b_2" : init_weights([32],"b_2"),
    "b_3" : init_weights([3],"b_3")
}

def model(X,weights,bias) :
    layer = tf.image.resize_bicubic(X,(200,200) , align_corners=None, name=None)
    layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(layer,weights["w_1"], strides=[1,1,1,1], padding="VALID"),bias["b_1"]))
    layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(layer,weights["w_2"], strides=[1,1,1,1], padding="VALID"),bias["b_2"]))
    layer = tf.nn.bias_add(tf.nn.conv2d(layer,weights["w_3"], strides=[1,1,1,1], padding="VALID"),bias["b_3"],name="layer")
    
    return layer

epoch = #number of epoch
batch_interation = #number of iteration in one epoch

saver = tf.train.Saver()

train_mse = []
test_mse = []

# train_mse = pickle.load(open('sess/srcnn/train_mse-multiple_scale.pkl', 'r'))
# test_mse = pickle.load(open('sess/srcnn/test_mse-multiple_scale.pkl', 'r'))


super_resolution = model(X,weights,bias)
cost = tf.reduce_mean(tf.losses.mean_squared_error(Y,super_resolution))
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

init = tf.global_variables_initializer()

low_r_size_list = [(100,100),(66,66),(50,50)]

with tf.Session() as sess:
    sess.run(init)
    #saver.restore(sess,"sess/srcnn/model.ckpt")
    
    for i in range(epoch) :
        gs += 1
        cost_sum = 0
        
        for j in range(batch_interation) :
            x , y,scale_factor = get_train_batch(40)
            
            low_r_size = low_r_size_list[scale_factor]
            
            x = x.reshape(-1,low_r_size[0],low_r_size[1],3)
            x = x / float(255)
            y = y / float(255)
            
            _,c = sess.run([train_op,cost],feed_dict={X:x,Y:y,global_step:gs})
            cost_sum += c
            
        print "epoch : " + str(i)
        print "train MSE : " + str(cost_sum/float(batch_interation))
        
        c = 0
        for k in range(50) :
            x , y, scale_factor = get_test_batch(100)
            
            low_r_size = low_r_size_list[scale_factor]
            
            x = x.reshape(-1,low_r_size[0],low_r_size[1],3)
            x = x / float(255)
            y = y / float(255)

            c += sess.run(cost,feed_dict={X:x,Y:y})
        
        print "test MSE : " + str(c/float(50))
        
        train_mse.append(cost_sum/float(batch_interation))
        test_mse.append(c)
    
        saver.save(sess, "sess/srcnn/model-multiple_scale.ckpt")
        with open('sess/srcnn/train_mse-multiple_scale.pkl', 'w') as f:
            pickle.dump(train_mse, f)
        with open('sess/srcnn/test_mse-multiple_scale.pkl', 'w') as f:
            pickle.dump(test_mse, f)

```

<h3>VDSR(<a>https://arxiv.org/abs/1511.04587</a>)</h3>

![](https://i.imgur.com/ySi0LLS.png)

VDSR(Very Deep Super Resolution) has much deeper CNN networks. All of convolution layers have 3 x 3 filters and its architecture is similar with VGG networks. To solve convergence of network problem, it employed large learning rate and residual learning (add input tensor and last convolutional layer). Single VDSR model can process multiple scales.

```python

X = tf.placeholder("float", [None, None,None,3],name="x")
Y = tf.placeholder("float", [None, 200,200,3],name="y")

weights = {
    "w_01" : init_weights([3,3,3,64],"w_01"),
    "w_20" : init_weights([3,3,64,3],"w_20")
}
bais = {
    "b_01" : init_weights([64],"b_01"),
    "b_20" : init_weights([3],"b_20")
}

for i in range(2,20) :
    weights["w_%02d" % (i)] = init_weights([3,3,64,64],"w_%02d" % (i))
    bais["b_%02d" % (i)] = init_weights([3,3,64,64],"b_%02d" % (i))


def model(X,weights,bias) :
    
    input_tensor = tf.image.resize_bicubic(X,(200,200) , align_corners=None, name=None)
    
    layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, weights['w_01'], strides=[1,1,1,1], padding='SAME'), bais['b_01']))
    
    for i in range(2,21) :
        layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, weights["w_%02d" % (i)], strides=[1,1,1,1], padding='SAME'), bais["b_%02d" % (i)]))
    
    layer = tf.add(layer,input_tensor)

    return layer

#training code is same with SRCNN

```
