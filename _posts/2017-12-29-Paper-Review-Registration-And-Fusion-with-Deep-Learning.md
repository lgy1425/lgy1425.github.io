---
layout: post
title:  "Paper Review : Medical Image Registration and Fusion with Deep Learning"
categories: Deep Learning
---

I had reviewed some papers for brain CT and MR registration and fusion with deep learning project. The number of researches that employed deep learning method for image registration and fusion is lower than other fields such as classification, segmentation, etc. But, I could get some insights from some papers. There are two main ways of employing CNN for registration and fusion. The first one is employing CNN regressor for spatial tranformation parameters. The second one is employing feature map extracted from CNN for silimarity metric. I will introduce 5 papers very shortly. If you want to get detail, you can get full-text from URLs.

<h4>1. End-to-End Unsupervised Deformable Image Registration with a Convolutional Neural Network</h4>
URL : <a href="https://arxiv.org/abs/1704.06065">https://arxiv.org/abs/1704.06065</a>
![](https://i.imgur.com/znPuMy9.png)

DIRNET(deformable image registration) can be trained to generate transformation parameters(dx,dy) at patch-wise. With tranformation parameters at each patch, spatial tranformer and resampler warped moving images. Back-propagation with similarity metric between warped images and fixed images optimized ConvNet regressor. DIRNet implementation in Tensorflow is produced at github page(https://github.com/iwyoo/DIRNet-tensorflow).

<h4>2. Fast Predictive Image Registration - a Deep Learning Approach</h4>
URL : <a href="https://arxiv.org/abs/1703.10908">https://arxiv.org/abs/1703.10908</a>

![](https://i.imgur.com/vBwiPat.png)
![](https://i.imgur.com/uiobJrC.png)
![](https://i.imgur.com/BpMdTUW.png)

There are 3 steps in this paper.
1. Train the prediction network using training images and the ground truth initial momentum obtained by numerical optimization of the LDDMM registration model.
2. Use the predicted momentum from the prediction network to generate deformation fields to warp the target images in the training dataset back to the space of the moving images.
3. Use the moving images and the warped-back target images to train the correction network. The correction network learns to predict the diffrence between the ground truth momentum and the predictied momentum from the prediction network.

In prediction LDDMM network, there are two split encoders and one decoder. It can make the network suitable to multi-modal input. And it has correction networks to finely tuning the tranformations.

<h4>3. A Medical Image Fusion Method Based on Convolutional Neural Networks</h4>

URL : <a href="http://ieeexplore.ieee.org/document/8009769/">http://ieeexplore.ieee.org/document/8009769/</a>

![](https://i.imgur.com/ege6Uyp.png)
![](https://i.imgur.com/vvH5dat.png)
![](https://i.imgur.com/UFQ7xDN.png)

It is fusion method with CNN. Two source images are converted to weight map through CNN. Two source images and weight map are fused and it is reconstructed by Laplacian pyramid.


<h4>4. Scalable High Performance Image Registration Framework by Unsupervised Deep Feature Representations Learning</h4>

![](https://i.imgur.com/Q4IKltR.png)

With stacked auto-encoder, critical points from feature representation are employed for registrations.

![](https://i.imgur.com/nigsH6Q.png) 

<h4>5. A CNN Regression Approach for Real-Time 2D/3D Registration</h4>

![](https://i.imgur.com/TbNmXjq.png)
![](https://i.imgur.com/wC52y1O.png)
![](https://i.imgur.com/uPo9dXs.png)

Patch-wise CNN regressors are trained to predict 6 tranformation parameters referred above. 