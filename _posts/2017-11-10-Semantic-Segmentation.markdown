---
layout: post
title:  "Image Semantic Segment Networks and Tensorflow implementations"
categories: deep-learning
---

Semantic segmentation using CNN is a more difficult task than image classification. This article briefly introduces the Fully Connected Networks(https://arxiv.org/abs/1605.06211), Deconvolution Networks(https://arxiv.org/abs/1505.04366), Dilated Convolution(https://arxiv.org/abs/1606.00915), and U-net(https://arxiv.org/abs/1505.04597) among the semantic segmentation methods using CNN and implements them with Tensorflow. For accurate semantic segmentation, it is necessary to learn both the overall context and detail of the image. In general, if you go through the pooling process a lot, you can see the entire image, but you can miss the details. The four networks to be introduced from now on have evolved to solve this problem.

<h3>Preparations</h3>

First we need to create a ground truth image. In this case, the inner and outer walls of the blood vessel should be distinguished from the ultrasound image. The figure below is an example.

![](https://i.imgur.com/vgEP7T7.png)

Labeling the inside out with lines as above uses the flood fill function. The code below uses a slightly modified flood fill method of the pillow package.

```python

def floodfill(image, xy, value, border=None, thresh=0):
    """
    (experimental) Fills a bounded region with a given color.
    :param image: Target image.
    :param xy: Seed position (a 2-item coordinate tuple).
    :param value: Fill color.
    :param border: Optional border value.  If given, the region consists of
        pixels with a color different from the border color.  If not given,
        the region consists of pixels having the same color as the seed
        pixel.
    :param thresh: Optional threshold value which specifies a maximum
        tolerable difference of a pixel value from the 'background' in
        order for it to be replaced. Useful for filling regions of non-
        homogeneous, but similar, colors.
    """
    # based on an implementation by Eric S. Raymond
    pixel = image.load()
    x, y = xy
    try:
        background = pixel[x, y]
        
        if _color_diff(value, background) <= thresh:
            return  # seed point already has fill color
        pixel[x, y] = value
    except (ValueError, IndexError):
        return  # seed point outside image
    edge = [(x, y)]
    if border is None:
        while edge:
            newedge = []
            for (x, y) in edge:
                for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                    try:
                        p = pixel[s, t]
                    except IndexError:
                        pass
                    else:
                        if _color_diff(p, background) <= thresh:
                            pixel[s, t] = value
                            newedge.append((s, t))
            edge = newedge
    else:
        while edge:
            newedge = []
            for (x, y) in edge:
                for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                    try:
                        p = pixel[s, t]
                    except IndexError:
                        pass
                    else:
                        if p != value and p != border:
                            pixel[s, t] = value
                            newedge.append((s, t))
            edge = newedge

            
def _color_diff(rgb1, rgb2):
    """
    Uses 1-norm distance to calculate difference between two rgb values.
    """
    return abs(rgb1[0]-rgb2[0]) +  abs(rgb1[1]-rgb2[1]) +  abs(rgb1[2]-rgb2[2])    

### example of generating annotation image

img = Image.open("segment_data/"+p+"/"+v+"/abnormal/"+f)
points = []
xs = []
ys = []
for x in range(0,480) :
    for y in range(0,480) :
        if isRed(img.getpixel((x,y))) :
            xs.append(x)
            ys.append(y)
            points.append((x,y))
            
mask = Image.new("RGB",((480,480)),(0,0,0))

draw = ImageDraw.Draw(mask)
for point in points :
    mask.putpixel(point,(255,255,255))

floodfill(mask,(1,1),(255,255,255),border=(255,255,255))
for point in points :
    mask.putpixel(point,(0,0,0))
    
    
mask = mask.convert("L")

mask = PIL.ImageOps.invert(mask)

for x in range(1,479) :
    for y in range(1,479) :
        if mask.getpixel((x-1,y-1)) > 200 and mask.getpixel((x-1,y)) > 200 and mask.getpixel((x-1,y+1)) > 200 and mask.getpixel((x,y-1)) > 200 and mask.getpixel((x,y+1)) > 200 and mask.getpixel((x,y+1)) > 200 and mask.getpixel((x+1,y-1)) > 200 and mask.getpixel((x+1,y+1)) > 200 :
            mask.putpixel((x,y),(255))
mask.save("annotation.jpg")    

```

The next step is to create a batch. Create a batch by creating the original image and label using the AnnotationToOneHot function, which converts the annotation to one hot vector. 'train_files' is list of file directories

```python

def AnnotationToOneHot(arr) :
    anno = np.zeros((arr.shape[0],arr.shape[1],2))
    black = 0
    white = 0
    for x in range(arr.shape[0]) :
        for y in range(arr.shape[1]) :
            if arr[x][y] < 128 :
                black += 1
                anno[x][y] = [1,0]
            else :
                white += 1
                anno[x][y] = [0,1]
    
    return anno

def train_batch(batch_size=10) :
    frames = np.random.choice(train_files,10,replace=False)
    origin = []
    annotation = []
    
    
    for f in frames :
        anno = Image.open(f['anno']).resize((width,height))
        img = Image.open(f['img']).resize((width,height))
        annotation.append(DenseToOneHot(np.array(anno)))
        origin.append(np.array(img))
    
    return np.array(origin)/ float(255), np.array(annotation)

```

The following are the functions related to training. Here the loss function is cross_entropy. Model() will be FCN, Deconvolution networks, Dilated Convolution, U-net models. Accuracy parameter is IoU(Intersection over Union)

```python

def get_iou(lin_pred,lin_y) :
    tp = tf.reduce_sum(tf.multiply(lin_y, lin_pred), 1)
    fn = tf.reduce_sum(tf.multiply(lin_y, 1-lin_pred), 1)
    fp = tf.reduce_sum(tf.multiply(1-lin_y, lin_pred), 1)
    return tf.reduce_mean((tp / (tp + fn + fp)))

pred = Model(x_, weights, biases, keepprob)
lin_pred = tf.reshape(pred, shape=[-1, 2])
lin_y = tf.reshape(y_, shape=[-1, 2])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = lin_pred, labels=lin_y))

predmax = tf.argmax(pred, 3,"predmax")
ymax = tf.argmax(y_, 3)

accr = get_iou(lin_pred,lin_y)
optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

```

<h3>FCN</h3>

![](https://i.imgur.com/chgCRm9.png)
![](https://i.imgur.com/ebkWjH4.png)

FCN is semantic segmentation using the activation map of general image classification by CNN. Like the general CNN structure, the activation map is compressed through the convolution layer and the pooling process. Compressed activation maps are upsampled by linear interpolation. Linear interpolation is not trainable. Implementation of FCN is below. Tensorflow can linear interpolation by tf.image.resize_images.


```python

weights = {
    'conv1_encoder': tf.get_variable("conv1_encoder", shape = [3, 3, 3, 64], initializer = initfun) ,
    'conv2_encoder': tf.get_variable("conv2_encoder", shape = [3, 3, 64, 64], initializer = initfun) ,
    'conv3_encoder': tf.get_variable("conv3_encoder", shape = [3, 3, 64, 128], initializer = initfun) ,
    'conv4_encoder': tf.get_variable("conv4_encoder", shape = [3, 3, 128, 128], initializer = initfun) ,
    'conv5_encoder': tf.get_variable("conv5_encoder", shape = [3, 3, 128, 256], initializer = initfun) ,
    'conv6_encoder': tf.get_variable("conv6_encoder", shape = [3, 3, 256, 256], initializer = initfun) ,
    'conv7_encoder': tf.get_variable("conv7_encoder", shape = [3, 3, 256, 256], initializer = initfun) ,
    
    'final_1x1': tf.get_variable("final_1x1", shape= [1, 1, 256, nrclass]
                                       , initializer = initfun) # <= 1x1conv
}
biases = {
    'b1': tf.get_variable("be1", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'b2': tf.get_variable("be2", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'b3': tf.get_variable("be3", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'b4': tf.get_variable("be4", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'b5': tf.get_variable("bd4", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'b6': tf.get_variable("bd3", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'b7': tf.get_variable("bd2", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'bf': tf.get_variable("bd1", shape = [256], initializer = tf.constant_initializer(value=0.0))
}

def Model(_X, _W, _b, _keepprob):

    encoder1 = tf.nn.conv2d(_X, _W['conv1_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder1 = tf.nn.bias_add(encoder1, _b['b1'])
           
    mean, var = tf.nn.moments(encoder1, [0, 1, 2])
    encoder1 = tf.nn.batch_normalization(encoder1, mean, var, 0, 1, 0.0001)
    
    encoder1 = tf.nn.relu(encoder1)
    
    encoder2 = tf.nn.conv2d(encoder1, _W['conv2_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder2 = tf.nn.bias_add(encoder2, _b['b2'])
           
    mean, var = tf.nn.moments(encoder2, [0, 1, 2])
    encoder2 = tf.nn.batch_normalization(encoder2, mean, var, 0, 1, 0.0001)
    
    encoder2 = tf.nn.relu(encoder2)
    
    encoder2 = tf.nn.max_pool(encoder2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    encoder1 = tf.nn.dropout(encoder1, _keepprob)
    
    #####################################################################################
    
    encoder3 = tf.nn.conv2d(encoder2, _W['conv3_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder3 = tf.nn.bias_add(encoder3, _b['b3'])
           
    mean, var = tf.nn.moments(encoder3, [0, 1, 2])
    encoder3 = tf.nn.batch_normalization(encoder3, mean, var, 0, 1, 0.0001)
    
    encoder3 = tf.nn.relu(encoder3)
    
    encoder4 = tf.nn.conv2d(encoder3, _W['conv4_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder4 = tf.nn.bias_add(encoder4, _b['b4'])
           
    mean, var = tf.nn.moments(encoder4, [0, 1, 2])
    encoder4 = tf.nn.batch_normalization(encoder4, mean, var, 0, 1, 0.0001)
    
    encoder4 = tf.nn.relu(encoder4)
    
    encoder4 = tf.nn.max_pool(encoder4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    encoder4 = tf.nn.dropout(encoder4, _keepprob)
    
    #######################################################################################
    
    encoder5 = tf.nn.conv2d(encoder4, _W['conv5_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder5 = tf.nn.bias_add(encoder5, _b['b5'])
           
    mean, var = tf.nn.moments(encoder5, [0, 1, 2])
    encoder5 = tf.nn.batch_normalization(encoder5, mean, var, 0, 1, 0.0001)
    
    encoder5 = tf.nn.relu(encoder5)
    
    encoder6 = tf.nn.conv2d(encoder5, _W['conv6_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder6 = tf.nn.bias_add(encoder6, _b['b6'])
           
    mean, var = tf.nn.moments(encoder6, [0, 1, 2])
    encoder6 = tf.nn.batch_normalization(encoder6, mean, var, 0, 1, 0.0001)
    
    encoder6 = tf.nn.relu(encoder6)
    
    encoder7 = tf.nn.conv2d(encoder6, _W['conv7_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder7 = tf.nn.bias_add(encoder6, _b['b7'])
           
    mean, var = tf.nn.moments(encoder7, [0, 1, 2])
    encoder7 = tf.nn.batch_normalization(encoder7, mean, var, 0, 1, 0.0001)
    
    encoder7 = tf.nn.relu(encoder7)
    
    encoder7 = tf.nn.max_pool(encoder7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    encoder7 = tf.nn.dropout(encoder7, _keepprob)
    
    #######################################################################################
    
    
    encoder7 = tf.image.resize_images(encoder7, [height, width])
    
    output = tf.nn.softmax(tf.nn.conv2d(encoder7, _W['final_1x1'], strides=[1, 1, 1, 1], padding='SAME'))
    
    return output
```

<h3>Deconvolution Networks</h3>
Bilinear interpolation used in FCN is not trainable. So, the higher magnification of upsampling by bilinear interpolation, segmentation is more inaccurate like below. 

![](https://i.imgur.com/Ov6Bbl7.png)

To solve this problems, Deconvolution Networks use deconvolution layer in upsampling networks.

![](https://i.imgur.com/ZP2S6aX.png)
![](https://i.imgur.com/I1DovYs.png)

Implementation of Deconvolution Networks is below. Deconvolution layer is implemented by tf.nn.conv2d_transpose function in Tensorflow and you should use tf.nn.max_pool_with_argmax when doing pooling in endcoding layers to get argmax.
```python

weights = {
    'conv1_encoder': tf.get_variable("conv1_encoder", shape = [3, 3, 3, 64], initializer = initfun) ,
    'conv2_encoder': tf.get_variable("conv2_encoder", shape = [3, 3, 64, 64], initializer = initfun) ,
    'conv3_encoder': tf.get_variable("conv3_encoder", shape = [3, 3, 64, 128], initializer = initfun) ,
    'conv4_encoder': tf.get_variable("conv4_encoder", shape = [3, 3, 128, 128], initializer = initfun) ,
    'conv5_encoder': tf.get_variable("conv5_encoder", shape = [3, 3, 128, 256], initializer = initfun) ,
    'conv6_encoder': tf.get_variable("conv6_encoder", shape = [3, 3, 256, 256], initializer = initfun) ,
    'conv7_encoder': tf.get_variable("conv7_encoder", shape = [3, 3, 256, 256], initializer = initfun) ,
    'conv8_encoder': tf.get_variable("conv8_encoder", shape = [3, 3, 256, 512], initializer = initfun) ,
    'conv9_encoder': tf.get_variable("conv9_encoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv10_encoder': tf.get_variable("conv10_encoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv11_encoder': tf.get_variable("conv11_encoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv12_encoder': tf.get_variable("conv12_encoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv13_encoder': tf.get_variable("conv13_encoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    
    "fc1" : tf.get_variable("fc1", shape = [4, 8, 512, 4096], initializer = initfun) ,
    "fc2" : tf.get_variable("fc2", shape = [1, 1, 4096, 4096], initializer = initfun) ,
    "fc3" : tf.get_variable("fc3", shape = [4, 8, 512, 4096], initializer = initfun) ,
    
    'conv13_decoder': tf.get_variable("conv13_decoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv12_decoder': tf.get_variable("conv12_decoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv11_decoder': tf.get_variable("conv11_decoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv10_decoder': tf.get_variable("conv10_decoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv9_decoder': tf.get_variable("conv9_decoder", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv8_decoder': tf.get_variable("conv8_decoder", shape = [3, 3, 256, 512], initializer = initfun) ,
    'conv7_decoder': tf.get_variable("conv7_decoder", shape = [3, 3, 256, 256], initializer = initfun) ,
    'conv6_decoder': tf.get_variable("conv6_decoder", shape = [3, 3, 256, 256], initializer = initfun) ,
    'conv5_decoder': tf.get_variable("conv5_decoder", shape = [3, 3, 128, 256], initializer = initfun) ,
    'conv4_decoder': tf.get_variable("conv4_decoder", shape = [3, 3, 128, 128], initializer = initfun) ,
    'conv3_decoder': tf.get_variable("conv3_decoder", shape = [3, 3, 64, 128], initializer = initfun) ,
    'conv2_decoder': tf.get_variable("conv2_decoder", shape = [3, 3, 64, 64], initializer = initfun) ,
    'conv1_decoder': tf.get_variable("conv1_decoder", shape = [3, 3, 64, 64], initializer = initfun) ,

    
    'dense_inner_prod': tf.get_variable("dense_inner_prod", shape= [1, 1, 64, nrclass]
                                       , initializer = initfun) # <= 1x1conv
}
biases = {
    'be1': tf.get_variable("be1", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'be2': tf.get_variable("be2", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'be3': tf.get_variable("be3", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'be4': tf.get_variable("be4", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'be5': tf.get_variable("be5", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'be6': tf.get_variable("be6", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'be7': tf.get_variable("be7", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'be8': tf.get_variable("be8", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'be9': tf.get_variable("be9", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'be10': tf.get_variable("be10", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'be11': tf.get_variable("be11", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'be12': tf.get_variable("be12", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'be13': tf.get_variable("be13", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    
    'bfc3': tf.get_variable("bfc3", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    
    'bd13': tf.get_variable("bd13", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'bd12': tf.get_variable("bd12", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'bd11': tf.get_variable("bd11", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'bd10': tf.get_variable("bd10", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'bd9': tf.get_variable("bd9", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'bd8': tf.get_variable("bd8", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'bd7': tf.get_variable("bd7", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'bd6': tf.get_variable("bd6", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'bd5': tf.get_variable("bd5", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'bd4': tf.get_variable("bd4", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'bd3': tf.get_variable("bd3", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'bd2': tf.get_variable("bd2", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'bd1': tf.get_variable("bd1", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    
    
    'bdp': tf.get_variable("bdp", shape = [64], initializer = tf.constant_initializer(value=0.0))
}

def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)    
    
    
def unpool_layer2x2_batch(x, argmax):
    '''
    Args:
        x: 4D tensor of shape [batch_size x height x width x channels]
        argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
        values chosen for each output.
    Return:
        4D output tensor of shape [batch_size x 2*height x 2*width x channels]
    '''
    x_shape = tf.shape(x)
    out_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]]

    batch_size = out_shape[0]
    height = out_shape[1]
    width = out_shape[2]
    channels = out_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat([t2, t3, t1], 4)
    indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

    x1 = tf.transpose(x, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))


# DeconvNet Model
def Model(_X, _W, _b, _keepprob):

    encoder1 = tf.nn.conv2d(_X, _W['conv1_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder1 = tf.nn.bias_add(encoder1, _b['be1'])
           
    mean, var = tf.nn.moments(encoder1, [0, 1, 2])
    encoder1 = tf.nn.batch_normalization(encoder1, mean, var, 0, 1, 0.0001)
    
    encoder1 = tf.nn.relu(encoder1)
    
    encoder2 = tf.nn.conv2d(encoder1, _W['conv2_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder2 = tf.nn.bias_add(encoder2, _b['be2'])
           
    mean, var = tf.nn.moments(encoder2, [0, 1, 2])
    encoder2 = tf.nn.batch_normalization(encoder2, mean, var, 0, 1, 0.0001)
    
    encoder2 = tf.nn.relu(encoder2)
    
    encoder2,encoder2_argmax = tf.nn.max_pool_with_argmax(encoder2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    encoder1 = tf.nn.dropout(encoder1, _keepprob)
    
    #####################################################################################
    
    encoder3 = tf.nn.conv2d(encoder2, _W['conv3_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder3 = tf.nn.bias_add(encoder3, _b['be3'])
           
    mean, var = tf.nn.moments(encoder3, [0, 1, 2])
    encoder3 = tf.nn.batch_normalization(encoder3, mean, var, 0, 1, 0.0001)
    
    encoder3 = tf.nn.relu(encoder3)
    
    encoder4 = tf.nn.conv2d(encoder3, _W['conv4_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder4 = tf.nn.bias_add(encoder4, _b['be4'])
           
    mean, var = tf.nn.moments(encoder4, [0, 1, 2])
    encoder4 = tf.nn.batch_normalization(encoder4, mean, var, 0, 1, 0.0001)
    
    encoder4 = tf.nn.relu(encoder4)
    
    encoder4,encoder4_argmax = tf.nn.max_pool_with_argmax(encoder4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    encoder4 = tf.nn.dropout(encoder4, _keepprob)
    
    #######################################################################################
    
    encoder5 = tf.nn.conv2d(encoder4, _W['conv5_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder5 = tf.nn.bias_add(encoder5, _b['be5'])
           
    mean, var = tf.nn.moments(encoder5, [0, 1, 2])
    encoder5 = tf.nn.batch_normalization(encoder5, mean, var, 0, 1, 0.0001)
    
    encoder5 = tf.nn.relu(encoder5)
    
    encoder6 = tf.nn.conv2d(encoder5, _W['conv6_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder6 = tf.nn.bias_add(encoder6, _b['be6'])
           
    mean, var = tf.nn.moments(encoder6, [0, 1, 2])
    encoder6 = tf.nn.batch_normalization(encoder6, mean, var, 0, 1, 0.0001)
    
    encoder6 = tf.nn.relu(encoder6)
    
    encoder7 = tf.nn.conv2d(encoder6, _W['conv7_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder7 = tf.nn.bias_add(encoder6, _b['be7'])
           
    mean, var = tf.nn.moments(encoder7, [0, 1, 2])
    encoder7 = tf.nn.batch_normalization(encoder7, mean, var, 0, 1, 0.0001)
    
    encoder7 = tf.nn.relu(encoder7)
    
    encoder7,encoder7_argmax = tf.nn.max_pool_with_argmax(encoder7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    encoder7 = tf.nn.dropout(encoder7, _keepprob)
    
    #######################################################################################
    
    
    
    
    encoder8 = tf.nn.conv2d(encoder7, _W['conv8_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder8 = tf.nn.bias_add(encoder8, _b['be8'])
           
    mean, var = tf.nn.moments(encoder8, [0, 1, 2])
    encoder8 = tf.nn.batch_normalization(encoder8, mean, var, 0, 1, 0.0001)
    
    encoder8 = tf.nn.relu(encoder8)
    
    encoder9 = tf.nn.conv2d(encoder8, _W['conv9_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder9 = tf.nn.bias_add(encoder9, _b['be9'])
           
    mean, var = tf.nn.moments(encoder9, [0, 1, 2])
    encoder9 = tf.nn.batch_normalization(encoder9, mean, var, 0, 1, 0.0001)
    
    encoder9 = tf.nn.relu(encoder9)
    
    encoder10 = tf.nn.conv2d(encoder9, _W['conv10_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder10 = tf.nn.bias_add(encoder10, _b['be10'])
           
    mean, var = tf.nn.moments(encoder10, [0, 1, 2])
    encoder10 = tf.nn.batch_normalization(encoder10, mean, var, 0, 1, 0.0001)
    
    encoder10 = tf.nn.relu(encoder10)
    
    encoder10,encoder10_argmax = tf.nn.max_pool_with_argmax(encoder10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    encoder10 = tf.nn.dropout(encoder10, _keepprob)
    
    #######################################################################################
    
    encoder11 = tf.nn.conv2d(encoder10, _W['conv11_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder11 = tf.nn.bias_add(encoder11, _b['be11'])
           
    mean, var = tf.nn.moments(encoder11, [0, 1, 2])
    encoder11 = tf.nn.batch_normalization(encoder11, mean, var, 0, 1, 0.0001)
    
    encoder11 = tf.nn.relu(encoder11)
    
    encoder12 = tf.nn.conv2d(encoder11, _W['conv12_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder12 = tf.nn.bias_add(encoder12, _b['be12'])
           
    mean, var = tf.nn.moments(encoder12, [0, 1, 2])
    encoder12 = tf.nn.batch_normalization(encoder12, mean, var, 0, 1, 0.0001)
    
    encoder12 = tf.nn.relu(encoder12)
    
    encoder13 = tf.nn.conv2d(encoder12, _W['conv13_encoder'], strides=[1, 1, 1, 1], padding='SAME')
    encoder13 = tf.nn.bias_add(encoder13, _b['be13'])
           
    mean, var = tf.nn.moments(encoder13, [0, 1, 2])
    encoder13 = tf.nn.batch_normalization(encoder13, mean, var, 0, 1, 0.0001)
    
    encoder13 = tf.nn.relu(encoder13)
    
    encoder13,encoder13_argmax = tf.nn.max_pool_with_argmax(encoder13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    encoder13 = tf.nn.dropout(encoder13, _keepprob)
    
    #######################################################################################
    
    fc1 = tf.nn.conv2d(encoder13, _W['fc1'], strides=[1, 1, 1, 1], padding='SAME')
    fc2 = tf.nn.conv2d(fc1, _W['fc2'], strides=[1, 1, 1, 1], padding='SAME')
    
    
    fc3 = tf.nn.conv2d_transpose(fc2, _W['fc3'],tf.stack([tf.shape(_X)[0],4,8,512]) , [1, 1, 1, 1], padding="SAME") + _b['bfc3']

    #######################################################################################
    
    decoder13 = unpool_layer2x2_batch(fc3,encoder13_argmax)
    
    print decoder13.get_shape()
     
    decoder13 = tf.nn.conv2d_transpose(decoder13, _W['conv13_decoder'],tf.stack([tf.shape(_X)[0],height/16,width/16,512]) , [1, 1, 1, 1], padding="SAME")
    decoder13 = tf.nn.bias_add(decoder13, _b['bd13'])
           
    mean, var = tf.nn.moments(decoder13, [0, 1, 2])
    decoder13 = tf.nn.batch_normalization(decoder13, mean, var, 0, 1, 0.0001)
    
    decoder13 = tf.nn.relu(decoder13)
    
    decoder12 = tf.nn.conv2d_transpose(decoder13, _W['conv12_decoder'],tf.stack([tf.shape(_X)[0],height/16,width/16,512]) , [1, 1, 1, 1], padding="SAME")
    encoder12 = tf.nn.bias_add(decoder12, _b['bd12'])
           
    mean, var = tf.nn.moments(decoder12, [0, 1, 2])
    decoder12 = tf.nn.batch_normalization(decoder12, mean, var, 0, 1, 0.0001)
    
    decoder12 = tf.nn.relu(decoder12)
    
    decoder11 = tf.nn.conv2d_transpose(decoder12, _W['conv11_decoder'],tf.stack([tf.shape(_X)[0],height/16,width/16,512]) , [1, 1, 1, 1], padding="SAME")
    decoder11 = tf.nn.bias_add(decoder11, _b['bd11'])
           
    mean, var = tf.nn.moments(decoder11, [0, 1, 2])
    decoder11 = tf.nn.batch_normalization(decoder11, mean, var, 0, 1, 0.0001)
    
    decoder11 = tf.nn.relu(decoder11)
    
    decoder10 = unpool_layer2x2_batch(decoder11,encoder10_argmax)
    
    ###################################################################################
    
    decoder10 = tf.nn.conv2d_transpose(decoder10, _W['conv10_decoder'],tf.stack([tf.shape(_X)[0],height/8,width/8,512]) , [1, 1, 1, 1], padding="SAME")
    encoder10 = tf.nn.bias_add(decoder10, _b['bd10'])
           
    mean, var = tf.nn.moments(decoder10, [0, 1, 2])
    decoder10 = tf.nn.batch_normalization(decoder10, mean, var, 0, 1, 0.0001)
    
    decoder10 = tf.nn.relu(decoder10)
    
    decoder9 = tf.nn.conv2d_transpose(decoder10, _W['conv9_decoder'],tf.stack([tf.shape(_X)[0],height/8,width/8,512]) , [1, 1, 1, 1], padding="SAME")
    decoder9 = tf.nn.bias_add(decoder9, _b['bd9'])
           
    mean, var = tf.nn.moments(decoder9, [0, 1, 2])
    decoder9 = tf.nn.batch_normalization(decoder9, mean, var, 0, 1, 0.0001)
    
    decoder9 = tf.nn.relu(decoder9)
    
    decoder8 = tf.nn.conv2d_transpose(decoder9, _W['conv8_decoder'],tf.stack([tf.shape(_X)[0],height/8,width/8,256]) , [1, 1, 1, 1], padding="SAME")
    decoder8 = tf.nn.bias_add(decoder8, _b['bd8'])
           
    mean, var = tf.nn.moments(decoder8, [0, 1, 2])
    decoder8 = tf.nn.batch_normalization(decoder8, mean, var, 0, 1, 0.0001)
    
    decoder8 = tf.nn.relu(decoder8)
    
    decoder7 = unpool_layer2x2_batch(decoder8,encoder7_argmax)
    
    ###################################################################################
    
    decoder7 = tf.nn.conv2d_transpose(decoder7, _W['conv7_decoder'],tf.stack([tf.shape(_X)[0],height/4,width/4,256]) , [1, 1, 1, 1], padding="SAME")
    encoder7 = tf.nn.bias_add(decoder7, _b['bd7'])
           
    mean, var = tf.nn.moments(decoder7, [0, 1, 2])
    decoder7 = tf.nn.batch_normalization(decoder7, mean, var, 0, 1, 0.0001)
    
    decoder7 = tf.nn.relu(decoder7)
    
    decoder6 = tf.nn.conv2d_transpose(decoder7, _W['conv6_decoder'],tf.stack([tf.shape(_X)[0],height/4,width/4,256]) , [1, 1, 1, 1], padding="SAME")
    decoder6 = tf.nn.bias_add(decoder6, _b['bd6'])
           
    mean, var = tf.nn.moments(decoder6, [0, 1, 2])
    decoder6 = tf.nn.batch_normalization(decoder6, mean, var, 0, 1, 0.0001)
    
    decoder6 = tf.nn.relu(decoder6)
    
    decoder5 = tf.nn.conv2d_transpose(decoder6, _W['conv5_decoder'],tf.stack([tf.shape(_X)[0],height/4,width/4,128]) , [1, 1, 1, 1], padding="SAME")
    decoder5 = tf.nn.bias_add(decoder5, _b['bd5'])
           
    mean, var = tf.nn.moments(decoder5, [0, 1, 2])
    decoder5 = tf.nn.batch_normalization(decoder5, mean, var, 0, 1, 0.0001)
    
    decoder5 = tf.nn.relu(decoder5)
    
    decoder4 = unpool_layer2x2_batch(decoder5,encoder4_argmax)
    
    ###################################################################################
    
    decoder4 = tf.nn.conv2d_transpose(decoder4, _W['conv4_decoder'],tf.stack([tf.shape(_X)[0],height/2,width/2,128]) , [1, 1, 1, 1], padding="SAME")
    decoder4 = tf.nn.bias_add(decoder4, _b['bd4'])
           
    mean, var = tf.nn.moments(decoder4, [0, 1, 2])
    decoder4 = tf.nn.batch_normalization(decoder4, mean, var, 0, 1, 0.0001)
    
    decoder4 = tf.nn.relu(decoder4)
    
    decoder3 = tf.nn.conv2d_transpose(decoder4, _W['conv3_decoder'],tf.stack([tf.shape(_X)[0],height/2,width/2,64]) , [1, 1, 1, 1], padding="SAME")
    decoder3 = tf.nn.bias_add(decoder3, _b['bd3'])
           
    mean, var = tf.nn.moments(decoder3, [0, 1, 2])
    decoder3 = tf.nn.batch_normalization(decoder3, mean, var, 0, 1, 0.0001)
    
    decoder3 = tf.nn.relu(decoder3)
    
    decoder2 = unpool_layer2x2_batch(decoder3,encoder2_argmax)
    
    ###################################################################################
    
    decoder2 = tf.nn.conv2d_transpose(decoder2, _W['conv2_decoder'],tf.stack([tf.shape(_X)[0],height,width,64]) , [1, 1, 1, 1], padding="SAME")
    decoder2 = tf.nn.bias_add(decoder2, _b['bd2'])
           
    mean, var = tf.nn.moments(decoder2, [0, 1, 2])
    decoder2 = tf.nn.batch_normalization(decoder2, mean, var, 0, 1, 0.0001)
    
    decoder2 = tf.nn.relu(decoder2)
    
    decoder1 = tf.nn.conv2d_transpose(decoder2, _W['conv1_decoder'],tf.stack([tf.shape(_X)[0],height,width,64]) , [1, 1, 1, 1], padding="SAME")
    decoder1 = tf.nn.bias_add(decoder1, _b['bd1'])
           
    mean, var = tf.nn.moments(decoder1, [0, 1, 2])
    decoder1 = tf.nn.batch_normalization(decoder1, mean, var, 0, 1, 0.0001)
    
    decoder1 = tf.nn.relu(decoder1)

    output = tf.nn.softmax(tf.nn.conv2d(decoder1, _W['dense_inner_prod'], strides=[1, 1, 1, 1], padding='SAME'))
    
    return output

```

![](https://i.imgur.com/PoumMpl.png)

Unpooling makes activation map sparse. With deconvolution layers, sparsed activation become dense. It can make deconvolution networks segment images accurately.

<h3>Dilated Convolution</h3>

For accurate segmentation, large receptive field of convolution is needed. But, large filters cause a lot of computation. For solving this problem, dilated convolutions expand receptive fields. With dilated convolutions, convolution layer filters have large receptive field with same weights in filters. Rate is interval between weights in filters.

![](https://i.imgur.com/H6FH4OO.png)

In tensorflow, dilated convolution is implemented by tf.nn.atrous_conv2d function.

```python

weights = {
    'conv1_encoder': tf.get_variable("conv1_encoder", shape = [7, 7, 3, 64], initializer = initfun) ,
    'conv2_encoder': tf.get_variable("conv2_encoder", shape = [7, 7, 64, 64], initializer = initfun) ,
    'conv3_encoder': tf.get_variable("conv3_encoder", shape = [7, 7, 64, 128], initializer = initfun) ,
    'conv4_encoder': tf.get_variable("conv4_encoder", shape = [7, 7, 128, 128], initializer = initfun) ,
    
    'dense_inner_prod': tf.get_variable("dense_inner_prod", shape= [1, 1, 128, nrclass]
                                       , initializer = initfun) # <= 1x1conv
}
biases = {
    'b1': tf.get_variable("be1", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'b2': tf.get_variable("be2", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'b3': tf.get_variable("be3", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'b4': tf.get_variable("be4", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'bdp': tf.get_variable("bd1", shape = [256], initializer = tf.constant_initializer(value=0.0))
}

def Model(_X, _W, _b, _keepprob):

    encoder1 = tf.nn.atrous_conv2d(_X, _W['conv1_encoder'],rate=2, padding='SAME')
    encoder1 = tf.nn.bias_add(encoder1, _b['b1'])
           
    mean, var = tf.nn.moments(encoder1, [0, 1, 2])
    encoder1 = tf.nn.batch_normalization(encoder1, mean, var, 0, 1, 0.0001)
    
    encoder1 = tf.nn.relu(encoder1)
    
    encoder2 = tf.nn.atrous_conv2d(encoder1, _W['conv2_encoder'],rate=2, padding='SAME')
    encoder2 = tf.nn.bias_add(encoder2, _b['b2'])
           
    mean, var = tf.nn.moments(encoder1, [0, 1, 2])
    encoder2 = tf.nn.batch_normalization(encoder2, mean, var, 0, 1, 0.0001)
    
    encoder2 = tf.nn.relu(encoder2)
    
    encoder3 = tf.nn.atrous_conv2d(encoder2, _W['conv3_encoder'],rate=2, padding='SAME')
    encoder3 = tf.nn.bias_add(encoder3, _b['b3'])
           
    mean, var = tf.nn.moments(encoder3, [0, 1, 2])
    encoder3 = tf.nn.batch_normalization(encoder3, mean, var, 0, 1, 0.0001)
    
    encoder3 = tf.nn.relu(encoder3)
    
    encoder4 = tf.nn.atrous_conv2d(encoder3, _W['conv4_encoder'],rate=2, padding='SAME')
    encoder4 = tf.nn.bias_add(encoder4, _b['b4'])
           
    mean, var = tf.nn.moments(encoder4, [0, 1, 2])
    encoder4 = tf.nn.batch_normalization(encoder4, mean, var, 0, 1, 0.0001)
    
    encoder4 = tf.nn.relu(encoder4)
    
    output = tf.nn.softmax(tf.nn.conv2d(encoder4, _W['dense_inner_prod'], strides=[1, 1, 1, 1], padding='SAME'))
    
    print output.get_shape()
    
    return output

```

<h3>U-Net</h3>

U-net has a structure that concatenates activation maps of various sizes. With this structure, U-net can make possible accurate segmentation. Because large activation maps attribute to subtle segmentation and small activation maps attribute to overall segmentation. 

![](https://i.imgur.com/AuRH3KA.png)

Implementation of U-net with Tensorflow is below

```python

weights = {
    'conv1_1': tf.get_variable("conv1_1", shape = [3, 3, 3, 64], initializer = initfun) ,
    'conv1_2': tf.get_variable("conv1_2", shape = [3, 3, 64, 64], initializer = initfun) ,
    'conv1_3': tf.get_variable("conv1_3", shape = [3, 3, 128, 64], initializer = initfun) ,
    'conv1_4': tf.get_variable("conv1_4", shape = [3, 3, 64, 64], initializer = initfun) ,
    
    'conv2_1': tf.get_variable("conv2_1", shape = [3, 3, 64, 128], initializer = initfun) ,
    'conv2_2': tf.get_variable("conv2_2", shape = [3, 3, 128, 128], initializer = initfun) ,
    'conv2_3': tf.get_variable("conv2_3", shape = [3, 3, 256, 128], initializer = initfun) ,
    'conv2_4': tf.get_variable("conv2_4", shape = [3, 3, 128, 128], initializer = initfun) ,
    'conv2_de': tf.get_variable("conv2_de", shape = [3, 3, 64, 128], initializer = initfun) ,
    
    'conv3_1': tf.get_variable("conv3_1", shape = [3, 3, 128, 256], initializer = initfun) ,
    'conv3_2': tf.get_variable("conv3_2", shape = [3, 3, 256, 256], initializer = initfun) ,
    'conv3_3': tf.get_variable("conv3_3", shape = [3, 3, 512, 256], initializer = initfun) ,
    'conv3_4': tf.get_variable("conv3_4", shape = [3, 3, 256, 256], initializer = initfun) ,
    'conv3_de': tf.get_variable("conv3_de", shape = [3, 3, 128, 256], initializer = initfun) ,
    
    'conv4_1': tf.get_variable("conv4_1", shape = [3, 3, 256, 512], initializer = initfun) ,
    'conv4_2': tf.get_variable("conv4_2", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv4_3': tf.get_variable("conv4_3", shape = [3, 3, 1024, 512], initializer = initfun) ,
    'conv4_4': tf.get_variable("conv4_4", shape = [3, 3, 512, 512], initializer = initfun) ,
    'conv4_de': tf.get_variable("conv4_de", shape = [3, 3, 256, 512], initializer = initfun) ,
    
    'conv5_1': tf.get_variable("conv5_1", shape = [3, 3, 512, 1024], initializer = initfun) ,
    'conv5_2': tf.get_variable("conv5_2", shape = [3, 3, 1024, 1024], initializer = initfun) ,
    'conv5_de': tf.get_variable("conv5_de", shape = [3, 3, 512, 1024], initializer = initfun) ,
    
    'dense_inner_prod': tf.get_variable("dense_inner_prod", shape= [1, 1, 64, nrclass]
                                       , initializer = initfun) # <= 1x1conv
}
biases = {
    'b1_1': tf.get_variable("b1_1", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'b1_2': tf.get_variable("b1_2", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'b1_3': tf.get_variable("b1_3", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    'b1_4': tf.get_variable("b1_4", shape = [64], initializer = tf.constant_initializer(value=0.0)),
    
    'b2_1': tf.get_variable("b2_1", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'b2_2': tf.get_variable("b2_2", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'b2_3': tf.get_variable("b2_3", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    'b2_4': tf.get_variable("b2_4", shape = [128], initializer = tf.constant_initializer(value=0.0)),
    
    'b3_1': tf.get_variable("b3_1", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'b3_2': tf.get_variable("b3_2", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'b3_3': tf.get_variable("b3_3", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'b3_4': tf.get_variable("b3_4", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    
    'b4_1': tf.get_variable("b4_1", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'b4_2': tf.get_variable("b4_2", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'b4_3': tf.get_variable("b4_3", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    'b4_4': tf.get_variable("b4_4", shape = [512], initializer = tf.constant_initializer(value=0.0)),
    
    'b5_1': tf.get_variable("b5_1", shape = [1024], initializer = tf.constant_initializer(value=0.0)),
    'b5_2': tf.get_variable("b5_2", shape = [1024], initializer = tf.constant_initializer(value=0.0)),
    
    'b5': tf.get_variable("bd4", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'b6': tf.get_variable("bd3", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'b7': tf.get_variable("bd2", shape = [256], initializer = tf.constant_initializer(value=0.0)),
    'bdp': tf.get_variable("bd1", shape = [256], initializer = tf.constant_initializer(value=0.0))
}


def Model(_X, _W, _b, _keepprob):

    layer1 = tf.nn.conv2d(_X, _W['conv1_1'], strides=[1, 1, 1, 1], padding='SAME')
    layer1 = tf.nn.bias_add(layer1, _b['b1_1'])
           
    mean, var = tf.nn.moments(layer1, [0, 1, 2])
    layer1 = tf.nn.batch_normalization(layer1, mean, var, 0, 1, 0.0001)
    
    layer1 = tf.nn.relu(layer1)
    
    layer1 = tf.nn.conv2d(layer1, _W['conv1_2'], strides=[1, 1, 1, 1], padding='SAME')
    layer1 = tf.nn.bias_add(layer1, _b['b1_2'])
           
    mean, var = tf.nn.moments(layer1, [0, 1, 2])
    layer1 = tf.nn.batch_normalization(layer1, mean, var, 0, 1, 0.0001)
    
    layer1 = tf.nn.relu(layer1)
    
    #########################################################################################
    
    layer2 = tf.nn.max_pool(layer1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    layer2 = tf.nn.conv2d(layer2, _W['conv2_1'], strides=[1, 1, 1, 1], padding='SAME')
    layer2 = tf.nn.bias_add(layer2, _b['b2_1'])
           
    mean, var = tf.nn.moments(layer2, [0, 1, 2])
    layer2 = tf.nn.batch_normalization(layer2, mean, var, 0, 1, 0.0001)
    
    layer2 = tf.nn.relu(layer2)
    
    layer2 = tf.nn.conv2d(layer2, _W['conv2_2'], strides=[1, 1, 1, 1], padding='SAME')
    layer2 = tf.nn.bias_add(layer2, _b['b2_2'])
           
    mean, var = tf.nn.moments(layer2, [0, 1, 2])
    layer2 = tf.nn.batch_normalization(layer2, mean, var, 0, 1, 0.0001)
    
    layer2 = tf.nn.relu(layer2)
    
    ##########################################################################################
    
    layer3 = tf.nn.max_pool(layer2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    layer3 = tf.nn.conv2d(layer3, _W['conv3_1'], strides=[1, 1, 1, 1], padding='SAME')
    layer3 = tf.nn.bias_add(layer3, _b['b3_1'])
           
    mean, var = tf.nn.moments(layer3, [0, 1, 2])
    layer3 = tf.nn.batch_normalization(layer3, mean, var, 0, 1, 0.0001)
    
    layer3 = tf.nn.relu(layer3)
    
    layer3 = tf.nn.conv2d(layer3, _W['conv3_2'], strides=[1, 1, 1, 1], padding='SAME')
    layer3 = tf.nn.bias_add(layer3, _b['b3_2'])
           
    mean, var = tf.nn.moments(layer3, [0, 1, 2])
    layer3 = tf.nn.batch_normalization(layer3, mean, var, 0, 1, 0.0001)
    
    layer3 = tf.nn.relu(layer3)
    
    ##########################################################################################
    
    layer4 = tf.nn.max_pool(layer3,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    layer4 = tf.nn.conv2d(layer4, _W['conv4_1'], strides=[1, 1, 1, 1], padding='SAME')
    layer4 = tf.nn.bias_add(layer4, _b['b4_1'])
           
    mean, var = tf.nn.moments(layer4, [0, 1, 2])
    layer4 = tf.nn.batch_normalization(layer4, mean, var, 0, 1, 0.0001)
    
    layer4 = tf.nn.relu(layer4)
    
    layer4 = tf.nn.conv2d(layer4, _W['conv4_2'], strides=[1, 1, 1, 1], padding='SAME')
    layer4 = tf.nn.bias_add(layer4, _b['b4_2'])
           
    mean, var = tf.nn.moments(layer4, [0, 1, 2])
    layer4 = tf.nn.batch_normalization(layer4, mean, var, 0, 1, 0.0001)
    
    layer4 = tf.nn.relu(layer4)
    
    ##########################################################################################
    
    layer5 = tf.nn.max_pool(layer4,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    layer5 = tf.nn.conv2d(layer5, _W['conv5_1'], strides=[1, 1, 1, 1], padding='SAME')
    layer5 = tf.nn.bias_add(layer5, _b['b5_1'])
           
    mean, var = tf.nn.moments(layer5, [0, 1, 2])
    layer5 = tf.nn.batch_normalization(layer5, mean, var, 0, 1, 0.0001)
    
    layer5 = tf.nn.relu(layer5)
    
    layer5 = tf.nn.conv2d(layer5, _W['conv5_2'], strides=[1, 1, 1, 1], padding='SAME')
    layer5 = tf.nn.bias_add(layer5, _b['b5_2'])
           
    mean, var = tf.nn.moments(layer5, [0, 1, 2])
    layer5 = tf.nn.batch_normalization(layer5, mean, var, 0, 1, 0.0001)
    
    layer5 = tf.nn.relu(layer5)
    
    ##########################################################################################
    
    layer5 = tf.nn.conv2d_transpose(layer5, _W['conv5_de'],tf.stack([tf.shape(_X)[0],height/8,width/8,512]) , [1, 2, 2, 1], padding="SAME")
    
    layer4 = tf.concat([layer4,layer5],axis=3)
    
    layer4 = tf.nn.conv2d(layer4, _W['conv4_3'], strides=[1, 1, 1, 1], padding='SAME')
    layer4 = tf.nn.bias_add(layer4, _b['b4_3'])
           
    mean, var = tf.nn.moments(layer4, [0, 1, 2])
    layer4 = tf.nn.batch_normalization(layer4, mean, var, 0, 1, 0.0001)
    
    layer4 = tf.nn.relu(layer4)
    
    layer4 = tf.nn.conv2d(layer4, _W['conv4_4'], strides=[1, 1, 1, 1], padding='SAME')
    layer4 = tf.nn.bias_add(layer4, _b['b4_4'])
           
    mean, var = tf.nn.moments(layer4, [0, 1, 2])
    layer4 = tf.nn.batch_normalization(layer4, mean, var, 0, 1, 0.0001)
    
    layer4 = tf.nn.relu(layer4)
    
    ##########################################################################################
    
    layer4 = tf.nn.conv2d_transpose(layer4, _W['conv4_de'],tf.stack([tf.shape(_X)[0],height/4,width/4,256]) , [1, 2, 2, 1], padding="SAME")
    
    layer3 = tf.concat([layer3,layer4],axis=3)
    
    layer3 = tf.nn.conv2d(layer3, _W['conv3_3'], strides=[1, 1, 1, 1], padding='SAME')
    layer3 = tf.nn.bias_add(layer3, _b['b3_3'])
           
    mean, var = tf.nn.moments(layer3, [0, 1, 2])
    layer3 = tf.nn.batch_normalization(layer3, mean, var, 0, 1, 0.0001)
    
    layer3 = tf.nn.relu(layer3)
    
    layer3 = tf.nn.conv2d(layer3, _W['conv3_4'], strides=[1, 1, 1, 1], padding='SAME')
    layer3 = tf.nn.bias_add(layer3, _b['b3_4'])
           
    mean, var = tf.nn.moments(layer3, [0, 1, 2])
    layer3 = tf.nn.batch_normalization(layer3, mean, var, 0, 1, 0.0001)
    
    layer3 = tf.nn.relu(layer3)
    
    ##########################################################################################
    
    layer3 = tf.nn.conv2d_transpose(layer3, _W['conv3_de'],tf.stack([tf.shape(_X)[0],height/2,width/2,128]) , [1, 2, 2, 1], padding="SAME")
    
    layer2 = tf.concat([layer2,layer3],axis=3)
    
    layer2 = tf.nn.conv2d(layer2, _W['conv2_3'], strides=[1, 1, 1, 1], padding='SAME')
    layer2 = tf.nn.bias_add(layer2, _b['b2_3'])
           
    mean, var = tf.nn.moments(layer2, [0, 1, 2])
    layer2 = tf.nn.batch_normalization(layer2, mean, var, 0, 1, 0.0001)
    
    layer2 = tf.nn.relu(layer2)
    
    layer2 = tf.nn.conv2d(layer2, _W['conv2_4'], strides=[1, 1, 1, 1], padding='SAME')
    layer2 = tf.nn.bias_add(layer2, _b['b2_4'])
           
    mean, var = tf.nn.moments(layer2, [0, 1, 2])
    layer2 = tf.nn.batch_normalization(layer2, mean, var, 0, 1, 0.0001)
    
    layer2 = tf.nn.relu(layer2)
    
    ##########################################################################################
    
    layer2 = tf.nn.conv2d_transpose(layer2, _W['conv2_de'],tf.stack([tf.shape(_X)[0],height,width,64]) , [1, 2, 2, 1], padding="SAME")
    
    layer1 = tf.concat([layer1,layer2],axis=3)
    
    layer1 = tf.nn.conv2d(layer1, _W['conv1_3'], strides=[1, 1, 1, 1], padding='SAME')
    layer1 = tf.nn.bias_add(layer1, _b['b1_3'])
           
    mean, var = tf.nn.moments(layer1, [0, 1, 2])
    layer1 = tf.nn.batch_normalization(layer1, mean, var, 0, 1, 0.0001)
    
    layer1 = tf.nn.relu(layer1)
    
    layer1 = tf.nn.conv2d(layer1, _W['conv1_4'], strides=[1, 1, 1, 1], padding='SAME')
    layer1 = tf.nn.bias_add(layer1, _b['b1_4'])
           
    mean, var = tf.nn.moments(layer1, [0, 1, 2])
    layer1 = tf.nn.batch_normalization(layer1, mean, var, 0, 1, 0.0001)
    
    layer1 = tf.nn.relu(layer1)
    
    
    output = tf.nn.softmax(tf.nn.conv2d(layer1, _W['dense_inner_prod'], strides=[1, 1, 1, 1], padding='SAME'))
    
    return output


```





