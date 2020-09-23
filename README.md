<h1 align="center">
<p>Traffic Sign Localization and Recognition
</h1>
<h6 align="center">
<p>a two stage system that finds out traffic signs from a given image then recognizes their class
  
</h3>

<p align="center">
 <img alt="cover" src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/overview.jpg" height="60%" width="60%">
</p>


### How to run
open up console in the project directory and enter this 
```
python localizeAndRecognize.py --cascade CASCADE_PATH --model MODEL_PATH --image IMAGE_PATH
```
the arguments are optional. If you dont provide them then the default values, *cascde.xml*, *trainedModel/bestModel.h5* and *input.jpg* are used.
```
python localizeAndRecognize.py
```

### Required libraries
Tensorflow 2.0, OpenCV 3, Numpy, Matplotlib, sci-kit learn
```
pip install -r requirements.txt
```
### Implementation
#### Localization 
To find out the regions containing traffic signs we used a
well known machine learning technique called Haar Cascade
Classifier. 
We used <a href="https://amin-ahmadi.com/cascade-trainer-gui/">this GUI tool</a> to train our cascade classifier using 500 positive images (samples) i.e. images of traffic signs from GTRSB dataset and 500 negative samples i.e. images of random objects. The features learned are contained in the output *cascade.xml* which is used by *OpenCV* to find out the Region of Interests (ROI) that might contain traffic sign.

<p align="center">
  <img src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/detect.png" width="400" /></p>


#### Recognition
The ROIs are cropped and passed to a CNN implemented on tensorflow. We  used publicly available dataset German Traffic Sign Recognition Benchmark to train our model. GTSRB dataset  is  a  multi-category  classification  competition  held  at IJCNN  2011.  The  dataset  is  composed  of  50,000  images  in total and 43 classes. The model is trained end to end using Adam optimizer with a initial learning rate of *0.0001* and a learning rate decay of *0.0001/(numberof epoch×0.5)*. The model was trained for 50 epochs with mini batch size of 64.

<p align="center">
  <img src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/data.png" width="200" />
    <img src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/accuracy.png" width="200" />
    <img src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/loss(1).png" width="200" />
</p>

Detailed documentation of the project can be found <a href="https://github.com/Zedd1558/Traffic-Sign-Localization-and-Recognition/blob/master/Report_on_the_Project.pdf">here</a>.

### Remarks
This is an extension of <a href="https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/">this tutorial</a> made by *pyimagsearch*. In this tutorial he shows us how to classify cropped traffic sign image using tensorflow. We extend his work to build an object detection system by implementing both localization and recognition stages.

### Contributors
https://github.com/Farabi-shafkat

https://github.com/Zedd1558
