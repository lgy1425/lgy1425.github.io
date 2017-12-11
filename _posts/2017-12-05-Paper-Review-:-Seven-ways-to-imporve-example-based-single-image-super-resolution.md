---
layout: post
title:  "Paper Review : Seven ways to improve example-based single image super resolution"
categories: computer vision
---

For improving single image super resolution, just CNN is not sufficient. Pre/Post processing image can improve super resolution performances. The paper "Seven ways to improve example-based single image super resolution(<a>https://arxiv.org/abs/1511.02228</a>)", introduce some of method for super resolution improvement. Those are 1) Data augmentation, 2) Hierarchical search, 3) Back projection, 4) Cascading, 5) Enhanced prediction, 6) Self similarity, 7) Context reasoning. These method increase PSNR score from 0.05 to 0.27dB. This article will provide summary of these methods and implementation code.

![](https://i.imgur.com/bW7F77f.png)


<h4>1. Data Augmentation</h4>

Inspired by the image classification literature, data augmentation also can improve super resolution quality. Just simple rotating original image by 90',180',270' and flipping upside-down can improve super resolution. Data augmentation can be easily implemented by python Pillow package or opencv.

![](https://i.imgur.com/ByZfwfh.png) 

```python
from PIL import Image
img = Image.open("img/1000.jpg")

#rotation
rotation90 = img.rotate(90)
rotation180 = img.rotate(180)
rotation270 = img.rotate(270)

#flip
flip = img.transpose(Image.FLIP_TOP_BOTTOM)
```

<h4>2. Hierarchical search</h4>

Basically, there are two methodologies for Super resolution: the Example-based method and the Sparse coding method. The example-based method is to learn with the LR image and its corresponding HR image. Sparse coding method is a method of rendering super resolution image by inferring surrounding pixel values with only LR. The most representative method in the example-based method is A + (Adjusted Anchored Neighborhood Regression). A + is an advanced method in ANR (Anchored Neighborhood Regression), and it set anchors in LR and HR images and is learned to make LR images become HR images near anchors. In other words, it is a method to learn the filter that can be made HR by patches in the LR images.A lot of anchors can get a more accurate image, but it takes a long time. In this paper, it proposed hierarchical search for setting anchors. Number of Anchor decrease from N to sqrt{N} with hierarchical search using K-means. That means anchors with similar contexts are grouped together and aggregated. This method did not increase PSNR much, but it can reduce time consuming in encoding time.

![](https://i.imgur.com/s79S91Y.png)

<h4>3. Back Projection</h4>
Iterative back projection makes HR reconstruction consistent with the LR input and degration operators such as blur, downscaling, and down-sampling. This method also have structure that updates back projection kernel for registration degration operations. Algorithm is like below image.

![](https://i.imgur.com/i2VNrnC.png) 

<h4>4. Cascading</h4>
 It is very simple concept that LR images are restored to HR images with smaller factors not objective factors. As the magnification factor is decreased, the superresolution becomes more accurate, since the space of possible HR solutions for each LR patch and thus the ambiguity decreases. As number of stages in cascade increase, PSNR also increases as you see below table.

 ![](https://i.imgur.com/dWEMwHE.png)

<h4>5. Enhanced Prediction</h4>
It is also a simple method.  In
SR image rotations and flips should lead to the same HR results at pixel level. Therefore, we apply rotations and flips on the LR image to get a set of 8 LR images(original, rotation 90,180,270,flip, 90&flip, 180&flip, 270&flip), then apply the SR method on each, reverse the transformation on the HR outputs and average for the final HR result.

<h4>6. Self-similarities</h4>
Images have a repeating structure in the image. Self-similarities can enhance the performance of the SR using this advantage. In other words, it means not only the external dictionary of HR and LR, but also the internal dictionary in LR image can be employed.

![](https://i.imgur.com/HEhw19z.png)
![](https://i.imgur.com/QxexgOx.png)

<h4>7. Reasoning with context</h4>
Training clustered images having similar context can enhance performace of SR. For example, if you learn human images and natural scenes separately, you can get better performance.

![](https://i.imgur.com/N5zGk12.png)

<h4>Conclusion</h4>
 Applying only deep learning method has limitation to high performance of SR. Pre/post processing referred above can enhance the performance much. It will be most important to choose the appropriate method depending on the type of image to be learned. And for a better SR, you need knowledge of computer vision and linear algebra. 

