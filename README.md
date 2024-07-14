# TFM
Object Detection, transmission via MQTT and object classification

## Solution
<p align="center">
  <img src="https://github.com/user-attachments/assets/a629c440-3d8a-4da2-8663-8b07fa85d311" />
</p>

The provided tool used to analyze surveillance camera recordings follows the next steps:
1) Detecting objects in motion using object detection methods, which has noise related issues.
2) Send this objec from the facilities to the Control Center. Assuming reliable communications between facilities and the Control Center with very low latency, detected objects with the motion detection algorithm will be sent through MQTT so as to be classified
3) Identify the object with DL classification algorithms due to their ability to recognize complex patterns, with the added benefit of using TL models. The DL algorithm used is the CLIP model, which overcomes the limitation of predicting a fixed number of categories and performs zero-shot classification as, due to the lack of diversity in our data (real recording from these cameras), the possibility of using data augmentation techniques was rapidly eliminated.

## Object Detection
Comparison of three object detection methods: 
* Frame Differencing: simple, fast, and efficient, but prone to generating false detections due to noise.
* Optical Flow: more accurate but requires high computational resources.
* Background Subtraction using both MOG2 and KNN: more complex but with fewer false positives.

## MQTT
The Mosquitto broker is used for MQTT data transmission between installations and the Control Center due to its low latency and scalability, using Docker Compose and TLS.

## Object Classification
Regarding the Deep Learning models used, the VIT-L-14 model, despite having a high processing time, has much more solid results compared to the next best model with a shorter processing time in terms of accuracy (VIT-B-32). 
It is also notable that the VIT-H-14 model has a significant difference in processing time, making it impractical for low-latency requirements. 
Additionally, the RN50 model, despite being a reference in image classification, shows very poor results for this project.
