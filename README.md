<h1 align="center">
<p>Traffic Sign Localization and Recognition
</h1>
<h6 align="center">
<p>a two stage system that finds out traffic signs from a given image then recognizes their class
  
</h3>

<p align="center">
 <img alt="cover" src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/overview.jpg" height="50%" width="50%">
</p>


### How to run
open up console in the project directory and enter this 
```
python detectionPlusRecognition.py
```

### Implementation
#### Localization 
To find out the regions containing traffic signs we used a
well known machine learning technique called Haar Cascade
Classifier. 
We used <a href="https://amin-ahmadi.com/cascade-trainer-gui/">this GUI tool</a> to train our cascade classifier using 500 positive images (samples) i.e. images of traffic signs from GTRSB dataset and 500 negative samples i.e. images of random objects. The features learned are contained in the output *cascade.xml* which is used by *OpenCV* to find out the Region of Interests that might contain traffic sign.
#### Recognition
The ROIs are cropped and passed to a CNN implemented on tensorflow. We  used publicly available dataset German Traffic Sign Recognition Benchmark to train our model. GTSRBdataset  is  a  multi-category  classification  competition  held  at IJCNN  2011.  The  dataset  is  composed  of  50,000  images  intotal and 43 classes. 

![model](mode_architecture.jpg "model architecture") ![dataset](data.png "samples from dataset")

<h4 align="center">
<p>let's see an exmaple
</h4>
<p align="center">
 <img alt="editing" src="">
</p>
<h4 align="center">
<p>removes text quite well!
</h4>

### Required libraries
PyQt, Numpy, OpenCV3, qimage2ndarray

<p align="center">
 <img alt="editing" src="https://github.com/Zedd1558/traffic-sign-recognition-tutorial-code/blob/master/documentation/best_model_confusion_matrix.png">
</p>


