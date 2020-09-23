import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
import argparse
import os

ap = argparse.ArgumentParser()
defaultModelPath = os.path.join("trainedModel","bestModel.h5")
ap.add_argument("-m","--model", default = defaultModelPath, help="path to the model")
ap.add_argument("-i","--image", default = "input.jpg", help="path to the image")
ap.add_argument("-c","--cascade", default = "cascade.xml", help="for detecting traffic signs, contains haar features")
args = vars(ap.parse_args())


#cascade.xml contains features needed to find traffic signs
cascade = cv2.CascadeClassifier(args["cascade"])


img = cv2.imread(args["image"])
full_image = np.array(img)


#loading CNN model
print("[INFO] loading model....")
model = load_model(args["model"])
labelnames = open("signnames.csv").read().strip().split("\n")[1:]
labelnames = [l.split(",")[1] for l in labelnames]

#detection
boxes = cascade.detectMultiScale(img, scaleFactor = 1.01, minNeighbors = 7, minSize= (24,24), maxSize=(128,128)) 


#recognition and drawing boundary boxes on input image
for (x,y,w,h) in boxes:
    print(x,y,w,h)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cropped_image = full_image[ y:y+h,x:x+w, : ]
    cropped_image = transform.resize(cropped_image,(32,32))
    cropped_image = exposure.equalize_adapthist(cropped_image, clip_limit=0.1)
    cropped_image = cropped_image.astype("float")/255.0
    cropped_image = np.expand_dims(cropped_image, axis=0)
    preds = model.predict(cropped_image)
    j = preds.argmax(axis=1)[0]
    label = labelnames[j]
    print(" j:",j," max_pred:",preds.max(), " label:",label)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.45, (0, 255, 255), 2)
    
#saving output file
cv2.imwrite('output.jpg', img)
