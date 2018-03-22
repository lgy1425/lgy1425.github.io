---
layout: post
title:  "OCR in Documents with AWS Rekongnition"
categories: Deep Learning
---

 I builed OCR engine detecting digits and recongnizing them in the documents. It was not general OCR engine. It was just for constant style documents(Korean Health Care Test Result Page). Below is an example of documents taken by mobile phone camera.
<br>
 <img src="https://i.imgur.com/hzBAUl4.jpg" width="500" />
<br>

I should find where red boxes are and recongnize digits in boxes. I found red boxes with OpenCV and recongnized digits with AWS Rekongnition. AWS Rekongnition is a service that provide trained deep learning model api for object and scene detection, image moderation, facial analysis, celebrity recongition, face comparision, text in images with cheap price. The digits in my documents are printed digit characters, so it is proper assignment for "Text in images" service in Rekongnition.

But, when area that characters occupy is small, the accuracy of Rekongntion decreases. So preprocessing of images was needed before request APIs. I preprocessed documents like below.

![](https://i.imgur.com/kKR57W2.jpg)

And, here is demo of "text in images".

![](https://i.imgur.com/jTh0Vj6.png)

First, I found red boxes with color detection after convert images' color to HSV and find red colors. And I made red color mask and find contours in the mask. These processes are implemented with python OpenCV.

```python

img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,20,0])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([160,20,0])
upper_red = np.array([190,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
(_,contours,_) = cv2.findContours(output_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

```

With the coutours, I found where red boxes are and blocked other areas. Next, I requested Rekongnition API with processed images. It was implemented with boto3 packages.

```python
client=boto3.client('rekognition','us-east-1',aws_access_key_id="access key",aws_secret_access_key="secret key")

image = Image.open(tempor_file_dir)
stream = io.BytesIO()
image.save(stream,format="JPEG")
image_binary = stream.getvalue()

response = client.detect_text(Image={'Bytes':image_binary})

```

Below is one of response examples.

```json
{'Confidence': 93.33549499511719,
  'DetectedText': '45',
  'Geometry': {'BoundingBox': {'Height': 0.07344907522201538,
    'Left': 0.6636187434196472,
    'Top': 0.3876687288284302,
    'Width': 0.05838353559374809},
   'Polygon': [{'X': 0.6635841131210327, 'Y': 0.3846329152584076},
    {'X': 0.721190869808197, uY': 0.3850800395011902},
    {'X': 0.7211101055145264, 'Y': 0.46381857991218567},
    {'X': 0.6635033488273621, 'Y': 0.46337148547172546}]},
  'Id': 17,
  'ParentId': 6,
  'Type': u'WORD'},


```

It was not complicate OCR model, but I can build OCR engine easily with AWS Rekongnition service.