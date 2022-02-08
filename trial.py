from glob import glob
import math
import tkinter as tk
from tkinter import *
from tkinter import ttk
from turtle import width
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

from arduino import lock, buzzer

root = tk.Tk()
root.title("SOP Violation Detector")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()

root.geometry('%dx%d+0+0' % (w,h))
root.state('zoomed')

icon=PhotoImage(file='icon.png')
root.iconphoto(False, icon)

bottom = Label(root, text="Developed by Muhammad Anas Bin Ateeq | Ayesha Aslam | Tooba Izat", width=w, font="Arial")
bottom.pack(side=BOTTOM)

head=Label(root,text="SOP VIOLATION DETECTOR", font=('DejaVu Sans Mono', 20))
head.place(relx=.2, rely=.4, anchor="c")

f1=LabelFrame(root,bg="red")
f1.pack(side=RIGHT)
bgimg = PhotoImage(file="background.pbm")
l1=Label(f1,image=bgimg,width=800,height=600)
l1.pack()



def distance():

    # social distance code
    labelsPath = "./coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

    weightsPath = "./yolov3.weights"
    configPath = "./yolov3.cfg"

    print("Loading Machine Learning Model ...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    cap=cv2.VideoCapture(0)

    global perm
    perm = True

    while (perm == True):

        ret, image = cap.read()
        image=cv2.flip(image,1)
        image = imutils.resize(image, width=800)
        (H, W) = image.shape[:2]
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        # print("Prediction time/frame : {:.6f} seconds".format(end - start))
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.1 and classID == 0:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
        ind = []
        for i in range(0,len(classIDs)):
            if(classIDs[i]==0):
                ind.append(i)
        a = []
        b = []

        if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    a.append(x)
                    b.append(y)

        distance=[] 
        nsd = []
        for i in range(0,len(a)-1):
            for k in range(1,len(a)):
                if(k==i):
                    break
                else:
                    x_dist = (a[k] - a[i])
                    y_dist = (b[k] - b[i])
                    d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                    distance.append(d)
                    if(d <=350):
                        nsd.append(i)
                        nsd.append(k)
                    nsd = list(dict.fromkeys(nsd))
                    print(nsd)
        color = (0, 0, 255) 
        for i in nsd:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "Violation detected"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        color = (0, 255, 0) 
        if len(idxs) > 0:
            for i in idxs.flatten():
                if (i in nsd):
                    break
                else:
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = 'Normal'
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   

        img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img= ImageTk.PhotoImage(Image.fromarray(img))
        l1['image']=img
        head['text']="Social Distance Detector Activated"
        root.update()


def mask():

    def detectFaceMask(frame, faceNet, maskNet):
    	# grab the dimensions of the frame and then construct a blob
    	# from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    	# pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

    	# initialize our list of faces, their corresponding locations,
    	# and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

    	# loop over the detections
        for i in range(0, detections.shape[2]):
    		# extract the confidence (i.e., probability) associated with
    		# the detection
            confidence = detections[0, 0, i, 2]

    		# filter out weak detections by ensuring the confidence is
    		# greater than the minimum confidence
            if confidence > 0.5 :
    			# compute the (x, y)-coordinates of the bounding box for
    			# the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

    			# ensure the bounding boxes fall within the dimensions of
    			# the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))             
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

    			# add the face and bounding boxes to their respective
    			# lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    	# only make a predictions if at least one face was detected
        if len(faces) > 0:
    		# for faster inference we'll make batch predictions on *all*
    		# faces at the same time rather than one-by-one predictions
    		# in the above `for` loop
            preds = maskNet.predict(faces)

    	# return a 2-tuple of the face locations and their corresponding
    	# locations
        return (locs, preds)


    # load our serialized face detector model from disk
    print("-------- Loading face detector...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("-------- Loading face mask detector ...")
    maskNet = load_model("mask_detector.model")

    # initialize the video stream and allow the camera sensor to warm up
    print("-------- Starting Camera...")
    
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    global perm
    # global br
    perm = True
    # br = True

    # loop over the frames from the video stream
    while (perm == True):
    	# grab the frame from the threaded video stream and resize it
    	# to have a maximum width of 400 pixels
        frame = vs.read()
        frame=cv2.flip(frame,1)
        frame = imutils.resize(frame, width=800)

        try:
            # detect faces in the frame and determine if they are wearing a
    	    # face mask or not
        
            (locs, preds) = detectFaceMask(frame, faceNet, maskNet)

    	    # loop over the detected face locations and their corresponding
    	    # locations
            for (box, pred) in zip(locs, preds):
    	    	# unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

    	    	# determine the class label and color we'll use to draw
    	    	# the bounding box and text
                global is_on

                if(mask > withoutMask):
                    label = "Face Mask"
                    color = (0, 255, 0)
                    lock(0)
                    buzzer(0)
                    is_on == FALSE
            
                else:
                    label = "No Face Mask"
                    color = (0, 0, 255)
                    lock(1)
                    if(is_on==TRUE):
                        buzzer(1)
                    else:
                        buzzer(0)
                    
    	    	# include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

    	    	# display the label and bounding box rectangle on the output
    	    	# frame
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        except:
            # print("More than one person")
            errormsg = "The detector checks only single person for entry"
            cv2.putText(frame, errormsg, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)        

    	# show the output frame
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img= ImageTk.PhotoImage(Image.fromarray(img))
        l1['image']=img
        head['text']="Face Mask Detector Activated"
        root.update()



def stop():
    global perm
    perm = FALSE
    global is_on
    is_on = FALSE
    l1['image']=bgimg
    head['text']="SOP VIOLATION DETECTOR"

    

        
iconimg = PhotoImage(file="icon.png")
pic=Label(root,image=iconimg,width=250,height=250)
pic.place(relx=.2, rely=.2, anchor="c")


launch_btn=Button(root,text="Launch Social Distance Detector",width=30,height=4,command=distance, padx=20)
launch_btn.place(relx=.2, rely=.5, anchor="c")
mask_btn=Button(root,text="Launch Face Mask Detector",width=30,height=4,command=mask, padx=20)
mask_btn.place(relx=.2, rely=.6, anchor="c")
stop_btn=Button(root,text="stop",width=30,height=4,command=stop, padx=20)
stop_btn.place(relx=.2, rely=.7, anchor="c")

is_on = True

def switch():
    global is_on
     
    # Determine is on or off
    if is_on:
        on_button.config(image = off)
        # buzzlabel.config(text = "The Switch is Off",
                        # fg = "grey")
        is_on = False
    else:
        on_button.config(image = on)
        # buzzlabel.config(text = "The Switch is On", fg = "green")
        is_on = True
 
# Define Our Images
on = PhotoImage(file = "on.png")
off = PhotoImage(file = "off.png")
 
# Create A Button
on_button = Button(root, image = on, bd = 0, command = switch)
on_button.place(relx=.2, rely=.9, anchor="c")


root.mainloop()