import sys
import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from subprocess import call

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, 5*128)
cap.set(4, 5*128)
SIZE = 28
img_width, img_height = 28, 28

if K.image_data_format() == "channels first":
    input_shape = (1, img_width, img_height)
    first_dim = 0
    second_dim = 0
else:
    input_shape = (img_width, img_height, 1)
    first_dim = 0
    second_dim = 3

# Writes labels on images
def put_labels(frame, label, location = (20, 30)):
    cv2.putText(frame, label, location, font, fontScale = 0.5, color = (255, 255, 0), thickness = 1, lineType = cv2.LINE_AA)
    
def extract_digits(frame, rect, pad = 10):
    x, y, w, h = rect
    crop_digit = final_img[y-pad:y+h+pad, x-pad:x+w+pad]
    crop_digit = crop_digit / 255.0
    
    # Only looking at images that are somewhat big
    if crop_digit.shape[0] >= 32 and crop_digit.shape[1] >= 32:
        crop_digit = cv2.resize(crop_digit, (SIZE, SIZE))
    else:
        return
    return crop_digit

def img_to_mnist(frame, thresh = 90):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    gray_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize = 321, C=28)
    return gray_img

print("loading model")
model =  load_model("digit_classifier.mnist")

labels = dict(enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))

for i in range(10000):
    ret, frame = cap.read(0)
    
    final_img = img_to_mnist(frame)
    image_shown = frame
    
    _, contours, _ = cv2.findContours(final_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = [cv2.boundingRect(contour) for contour in contours]
    rects = [rect for rect in rects if rect[2] >= 3 and rect[3] >= 8]
    
    #draw rectangles and predict
    for rect in rects:
        x, y, w, h = rect
        
        if i >= 0:
            
            mnist_frame = extract_digits(frame, rect, pad = 15)
            
            if mnist_frame is not None:
                mnist_frame = np.expand_dims(mnist_frame, first_dim)
                mnist_frame = np.expand_dims(mnist_frame, second_dim)
                
                class_prediction = model.predict_classes(mnist_frame, verbose = False)[0]
                prediction = np.around(np.max(model.predict(mnist_frame, verbose = False)), 2)
                label = str(prediction)
                
                cv2.rectangle(image_shown, (x - 15, y - 15), (x + 15 + w, y + 15 + h), color = (255, 255, 0))
                
                label = labels[class_prediction]
                
                put_labels(image_shown, label, location = (rect[0], rect[1]))
                
    cv2.imshow('frame', image_shown)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break