---
layout: post
title:  "Detecting and Classifying Tumor in Lung CT Images (Asan Bigdata Contest 2nd Rank)"
categories: deep-learning
---

In January 2017, AMC(Asan Medical Center, Korea) and Microsoft Korea held medical bigdata analysis contest. This research introduced in this article is about one of the 5 projects in the contest and was ranked as 2nd prize. This article introduce shortly research’s method, process and results

<h3>1. Subject</h3>

This research did assignment that deep learning models detects lung tumors and diagnosing whether the tumors is benign or malignant from CT images taking human’s lung. Until now, only trained and skillful medical staffs can diagnosing lung cancers with CT images. But, trained deep learning models are expected to be able to easily and quickly diagnosing lung cancers. That will be able to reduce cost and time of diagnosing lung cancers in CT images.

<h3>2. Data Set</h3>

All of data set used in this research are produced by medical big data analysis contest hold by Asan Medical Center. Produced data was human body CT image around lung and ROI(Region of Interest) mask images made by professional medical staffs in medical center. Data format was dicom files. Training data set was 300 patients’ CT images and ROI mask, 93 patients had benign tumors and 207 patients had malignant tumors. Validation data set for evaluating participants of contest was 62 patients’ CT images and ROI mask, composed of 21 patients having benign tumors and 42 patients having malignant tumors.
All of CT images had 512 x 512px resolution. All pixel values are integers between 0 and 2048 and images had only one channel and are black-white images. Each patient had 50 – 70 frames and an interval of frames is about 5mm. ROI mask’s resolution was same with CT image’s and ROI mask is black-white image.

![](https://i.imgur.com/ytHwcku.png)
![](https://i.imgur.com/bABnfoV.png)

<h3>3. Method</h3>
Processes are implemented with python(2.7.13) language and python basic package were used. Some python packages and libraries such as pydicom for processing dicom format files, Pillow for processing images, and numpy for math and matrix operations. For deep learning implementation, Tensorflow(http://tensorflow.org) was used.
<h3>Sliding Window</h3>
Sliding window method was applied to this research for analyzing images. Sliding window method is that kernel(window) slides on whole images from end to end with constant stride and each kernel is analyzed. The reason that sliding window method was applied is that ROIs(tumors) occupied small area(up to width, height 50px) in CT images(512x512px size) and all of ROIs can be contained in one kernel. And CT images was not very high resolution images, so time consuming problem was not serious for sliding window method.

![](https://i.imgur.com/hO9Cwx6.png)

Patches from sliding window method is 50x50px size with 10px strides. About 25,000 patches are made in each CT image. Cropped patches are divided into 3 classes label using ROI mask data. All patches are divided into 3 class, Not ROI(patches not containing ROI), Benign ROI(patches containing benign ROI), Malignant ROI(patches containing malignant ROI). Training data was composed of these 3 class patches. From 300 patients’ CT images, about 3,000 Benign ROI patches, about 19,000 Malignant ROI patches, about 20,000 Not ROI patches randomly extracted from millions patches compose training data for deep learning.

![](https://i.imgur.com/ltzdrzM.png)

<h3>Data Augmentation</h3>
Training data extracted from sliding window method was composed of about 20,000 Not ROI , 3,000 Benign ROI, 19,000 Malignant ROI. Normally, unbalance between numbers of training data’ classes can generate overfitting to CNN models. So, the number of Benign ROI class data was augmented by data augmentation method. There are some usual data augmentation technique for images such as random crop, flip, rotation, blurring, elastic deformation, dropout, zoom in and so on. In this research, horizontal flip, zoom in, 30 degree rotation, 45 degree rotation, blurring were used as data augmentation technique because CT images had up and down side and deformation of tumor shape can adversely affect accuracy of diagnosing.

![](https://i.imgur.com/96HMRgj.png)

Due to data augmentation, numbers of 3 classes data(Not ROI, Benign ROI, Malignant ROI) were balanced. Each number of 3 classes is about 20,000 in training data set.

<h3>Network Modeling</h3>
A customized CNN model was modeled for classifying 50x50px size patches into 3 classes. Usually used CNN models are customized for CT images. In first time, original VGG 16 model was used for training CNN model, but model couldn’t properly learn CT images. It was too deeper to learning CT image’s features. If model is too deep compared to complexity of data, it is hard for model to learn data’s features. Each patch is 50x50px size and one channel black-white image and it is similar with MNIST dataset. For MNIST data set, very simple CNN model can get over 99% classification accuracy. So the model for classifying patches into 3 classes was shallower model than VGG model and has similar architecture with VGG mdoel.

![](https://i.imgur.com/4YePA3N.png)
![](https://i.imgur.com/hYM9poG.png)

As seen in architecture image above, after passing through 2 convolution layers, max pooling is operated. After that, first max pooling are followed by 3 convolution layers. The last convolution layer is followed by max pooling operation. And next layers are 2 fully connected layers. Soft max operation is applied to last fully connected layer and its output is probability for each class. All of convolution layers’ filters are 3x3px size and stride size is 1px. The first 2 convolution layers’ channel is 54, the last 3 convolution layers’ channel is 108. Two max pooling operations’ kernel is 2x2px size and stride size is 2px. After max pooling operation, feature map become half size feature map. After second max pooling operation, feature map’s size is 13x13px and the number of channel is 108. This feature map is flatten to vector and the vector is connected to 2 fully connected layers. The first fully connected layer’s elements’ number is 432, the last fully connected layer’s elements’ number is 3 and same with the number of classes. The last output vector with softmax operation is probability of each class. All of layers have ReLU(Rectified Linear Unit) operation as activation function. Detailed parameters is shown on below table.

![](https://i.imgur.com/9z7jAoA.png)

<h3>Preventing Overfitting</h3>

Overfitting means that trained deep learning models can’t be generalized for other data not trained and trained model can predict training data set, but can’t for validation data set. For preventing overfitting, one of the most popular techniques is data augmentation referred above. Balance between numbers of classes with data augmentation makes generalized CNN models. And some techniques that makes weights in CNN models be regularized are usually used for preventing overfitting. In this research, dropout are applied to CNN model for preventing overfitting. Dropout means that some of weights in CNN models are randomly dropped out in training process for preventing some of weights much bigger than other weights and regularizing weights. In CNN models in this research, dropout was applied at two max pool operations and first fully connected layer. The rate of dropout in convolution layer is 30% and the rate of dropout in fully connected layer is 50%.

<h3>Training</h3>

The CNN model made up above was trained with about 60,000 patches. Training process is minimizing cost defined as cross entropy between 3 classes labeling vectors(Not ROI [1,0,0], Benign ROI [0,1,0], Malignant [0,0,1]) and softmax result passed through the CNN model. AdamOptimizer was used as optimizer for training. The learning rate of training was constant and 0.0005. Training batches were 400 patches randomly extracted from training data set and an epoch was defined that CNN model was trained for all of 60,000 patches. After training CNN model in 200 epochs, cross entropy was converged to 0.55. Training CNN model with 200 epochs data set took about 14 hours.

![](https://i.imgur.com/KmhVCSy.png)

<h3>ROI detection with trained model</h3>

If trained CNN model can classify patches into 3 classes with over 90 % accuracy, next step is detecting region of tumors in CT images with trained model. As seen in Fig 11, firstly patches are extracted from CT image in 50x50px size and 10px size stride, same with preprocessing(slding window). Each patch is labeled into 3 classes : Not ROI : 0, Benign ROI : 1, Malignant ROI : 2. As seen in downside of Fig 12, these labels were labeled on a new image having same size with a CT image. In labeled the image, region where 1 or 2 labels are overlapped is detected as ROIs.

![](https://i.imgur.com/E1MIxCg.png)
![](https://i.imgur.com/qDoWVP4.png)

<h3>Benign/Malignant Classification with trained model</h3>

In the region detected as ROI, CNN model predict whether the ROIs is benign or malignant with majority voting among 1 or 2 labels. The threshold of majority voting can be adjusted according to sensitivity.

<h3>4. Result</h3>

The result of this research is from validation data set composed of 62 patients’ CT images and ROI mask. In ROI detection result, detected ROI is masked with red and compared to mask professional medical staffs mdae. Diagnosing tumor as benign or malignant was done with majority voting in CT images in one patient. Diagnosing a CT image took about 5 seconds in GPU hardware environment.

<h3>ROI Detection</h3>

![](https://i.imgur.com/SDb4lgN.png)

Trained model detected some region probably contains ROI and masked it with red color. As Examples seen in Fig 12, big enough ROIs were completely detected by CNN model, some small ROIs were missed or other region not containing ROI were recommended . But CNN model did not miss all of tumors in one patient’s CT frames.

<h3>Benign/Malignant Classification</h3>

Among 62 patients in validation data set, 57 patients are correctly classified into benign and malignant. Its accuracy was 92%. 3 benign patients were diagnosed as malignant, 2 malignant patients were diagnosed as benign. Sensitivity for malignant tumor was 0.95 and specificity for benign was 0.9. 

![](https://i.imgur.com/ElLc766.png)

<h3>5. Conclusion</h3>

The sliding window method employed in this research is not generally better than other CNN object detection methods such as RCNN based network and YOLO. But in environment that objects(ROIs) is so small, sliding window can be effective. The training data’s volume was too small to be applied to deep learning method. But sliding window method and data preprocessing could make it sufficient to be applied to CNN models. It is expected to get more accurate model with huge volume of data and balance between benign and malignant data. Also, for solving difficulties of detecting small ROI, variety of sliding window patch’s size will be applied next time.  



