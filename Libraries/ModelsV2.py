#General
import cv2
import sys
import numpy as np
import imp
import tensorflow as tf
import os

import threading
import datetime
import time
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Script para codigo terminado

#Probar rendimiento
timestart = [time.perf_counter() for i in range(5)]
def tic(number = 0):
    global timestart
    timestart[number] = time.perf_counter()
    
def toc(number = 0):
    global timestart
    return (time.perf_counter()-timestart[number])

class tictocclass():
    def __init__(self):
        self.tictocTimer = time.perf_counter()
        
    def tic(self):
        self.tictocTimer = time.perf_counter()
        
    def toc(self):
        return (time.perf_counter()-self.tictocTimer)

def flushcam(cam, Sdelay = 0.001):
    framesWithDelayCount = 0
    while (framesWithDelayCount <= 1):
        tic()
        cam.grab()
        delay = toc()
        if(delay > Sdelay):
            framesWithDelayCount += 1

def load_model_EfficientDet(path,automl = False):
    model = tf.saved_model.load(path)
    if automl:
        model = model.signatures['serving_default']
    return model

class Lite_Detector(tf.lite.Interpreter):
    def __init__(self,model_path="coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite",efficient=False):
        super().__init__(model_path)
        self.isEfficient = efficient
        self.allocate_tensors()
        self.input_details = self.get_input_details()
        self.output_details = self.get_output_details()
        self.labels = load_labels(True)
        self.threshhold = 0.6
        self.sub_128 = False
        self.div_128 = False
    
    def set_threshhold(self, threshhold):
        self.threshhold = threshhold
        
    def set_normalize(self, sub_128,div_128):
        self.sub_128 = sub_128
        self.div_128 = div_128
    
    def detect(self,image):
        if self.sub_128:
            image = image.copy()-128
        if self.div_128:
            image = image.copy()/128
        input_data = self.input_details[0]['dtype'](np.expand_dims(cv2.resize(image,tuple(self.input_details[0]['shape'][1:3])),axis=0))
        self.set_tensor(self.input_details[0]['index'], input_data)
        self.invoke()
        output_dict = {}
        if self.isEfficient:
            partdict = self.get_tensor(self.output_details[0]['index'])[0]
            output_dict['detection_classes'] = np.int8(partdict[:,6])
            output_dict['detection_boxes'] = np.int16(partdict[:,1:5])
            output_dict['detection_scores']= partdict[:,5]
        else:
            output_dict['detection_classes'] = np.int8(self.get_tensor(self.output_details[1]['index'])[0])+1
            output_dict['detection_boxes'] = self.get_tensor(self.output_details[0]['index'])[0]
            output_dict['detection_scores']= self.get_tensor(self.output_details[2]['index'])[0]
        
        self.output_dict = output_dict
        return output_dict
    
    def detectEfficientNet(self,image):
        input_data = self.input_details[0]['dtype'](np.expand_dims(cv2.resize(image,tuple(self.input_details[0]['shape'][1:3])),axis=0))
        self.set_tensor(self.input_details[0]['index'], input_data)
        self.invoke()
        partdict = modelLite.get_tensor(output_details[0]['index'])[0]
        output_dict = {}
        output_dict['detection_classes'] = np.int8(partdict[:,6])
        output_dict['detection_boxes'] = np.int16(partdict[:,1:5])
        output_dict['detection_scores']= partdict[:,5]
    
    def Draw(self,image):
        if self.isEfficient:
            outputFrame = detect_all_efficient_Lite(image, self.output_dict, self.labels, self.input_details[0]['shape'][1], self.threshhold)
        else:
            outputFrame = detect_all(image, self.output_dict, self.labels, self.threshhold)
        input_details[0]['shape'][1]
        return outputFrame
    
    def DetectandDraw(self,image):
        self.detect(image)
        if self.isEfficient:
            outputFrame = detect_all_efficient_Lite(image, self.output_dict, self.labels, self.input_details[0]['shape'][1], self.threshhold)
        else:
            outputFrame = detect_all(image, self.output_dict, self.labels, self.threshhold)
        return outputFrame
    
    def FilterbyClass(self,classes):
        assert type(classes) is list, 'classes shoud be a list'
        classfind = [self.output_dict['detection_classes']==i for i in classes]
        classinter = [any([p[i] for p in classfind]) for i in range(len(classfind[0]))]
        intersection = np.bitwise_and(self.output_dict['detection_scores']>self.threshhold,classinter)
        self.output_dict['detection_classes'] = self.output_dict['detection_classes'][intersection]
        self.output_dict['detection_boxes'] = self.output_dict['detection_boxes'][intersection]
        self.output_dict['detection_scores'] = self.output_dict['detection_scores'][intersection]


#Tomar salida de red y dibujar personas
def detect_people(image, output_dict, thresh = 0.50):
    imageBox = image.copy()
    h,w,_ = image.shape
    gooddetections = np.argwhere(output_dict['detection_scores']>thresh)
    rightcat = np.argwhere(output_dict['detection_classes']==1)
    intersects = np.intersect1d(gooddetections, rightcat)
    for i in intersects:
        y1,x1,y2,x2 = output_dict['detection_boxes'][i]
        x1,y1,x2,y2 = [int(imageBox.shape[1]*x1),int(imageBox.shape[0]*y1),int(imageBox.shape[1]*x2),int(imageBox.shape[0]*y2)]
        x1,y1,x2,y2 = max(x1,0),max(y1,0),min(x2,w),min(y2,h)
        cv2.rectangle(imageBox,(x1,y1),(x2,y2),(0,0,255),2)
        
        cv2.putText(imageBox,str(int(output_dict['detection_scores'][i]*100))+'%',
                    (x1, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(imageBox,str(int(output_dict['detection_scores'][i]*100))+'%',
                    (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return imageBox

def detect_cars(image, output_dict, thresh = 0.50):
    imageBox = image.copy()
    h,w,_ = image.shape
    gooddetections = np.argwhere(output_dict['detection_scores']>thresh)
    rightcat = np.argwhere(output_dict['detection_classes']==3)
    intersects = np.intersect1d(gooddetections, rightcat)
    for i in intersects:
        y1,x1,y2,x2 = output_dict['detection_boxes'][i]
        x1,y1,x2,y2 = [int(imageBox.shape[1]*x1),int(imageBox.shape[0]*y1),int(imageBox.shape[1]*x2),int(imageBox.shape[0]*y2)]
        x1,y1,x2,y2 = max(x1,0),max(y1,0),min(x2,w),min(y2,h)
        cv2.rectangle(imageBox,(x1,y1),(x2,y2),(0,0,255),2)
        
        cv2.putText(imageBox,str(int(output_dict['detection_scores'][i]*100))+'%',
                    (x1, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(imageBox,str(int(output_dict['detection_scores'][i]*100))+'%',
                    (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return imageBox

def getcrop(image, output_dict, thresh = 0.50, ythresh = 0.5):
    imageBox = image.copy()
    h,w,_ = image.shape
    gooddetections = np.argwhere(output_dict['detection_scores']>thresh)
    rightcat1 = np.argwhere(output_dict['detection_classes']==3)
    #rightcat2 = np.argwhere(output_dict['detection_classes']==6)
    rightcat3 = np.argwhere(output_dict['detection_classes']==8)
    rightcat = np.concatenate([rightcat1,rightcat3])
    intersects = np.intersect1d(gooddetections, rightcat)
    images = []
    for i in intersects:
        y1,x1,y2,x2 = output_dict['detection_boxes'][i]
        x1,y1,x2,y2 = [int(imageBox.shape[1]*x1),int(imageBox.shape[0]*y1),int(imageBox.shape[1]*x2),int(imageBox.shape[0]*y2)]
        if y1>ythresh*h:
            x1,y1,x2,y2 = max(x1,0),max(y1,0),min(x2,w),min(y2,h)
            images.append(imageBox[y1:y2,x1:x2])
    return images

#Tomar salida de red y dibujar objetos detectados
def detect_all(image, output_dict, labels_dict, thresh = 0.50):
    imageBox = image.copy()
    h,w,_ = image.shape
    gooddetections = np.argwhere(output_dict['detection_scores']>thresh).ravel()
    for i in gooddetections:
        y1,x1,y2,x2 = output_dict['detection_boxes'][i]
        x1,y1,x2,y2 = [int(imageBox.shape[1]*x1),int(imageBox.shape[0]*y1),int(imageBox.shape[1]*x2),int(imageBox.shape[0]*y2)]
        x1,y1,x2,y2 = max(x1,0),max(y1,0),min(x2,w),min(y2,h)
        cv2.rectangle(imageBox,(x1,y1),(x2,y2),(0,0,255),2)
        
        cv2.putText(imageBox,str(labels_dict[str(output_dict['detection_classes'][i])]) + ': ' + str(int(output_dict['detection_scores'][i]*100))+'%',
                    (x1, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(imageBox,str(labels_dict[str(output_dict['detection_classes'][i])]) + ': ' + str(int(output_dict['detection_scores'][i]*100))+'%',
                    (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return imageBox


def detect_all_efficient_Lite(image, output_dict, labels_dict, scale = 300, thresh = 0.50):
    imageBox = image.copy()
    h,w,_ = image.shape
    gooddetections = np.argwhere(output_dict['detection_scores']>thresh).ravel()
    for i in gooddetections:
        y1,x1,y2,x2 = output_dict['detection_boxes'][i]
        x1,y1,x2,y2 = [int(imageBox.shape[1]/scale*x1),int(imageBox.shape[0]/scale*y1),int(imageBox.shape[1]/scale*x2),int(imageBox.shape[0]/scale*y2)]
        x1,y1,x2,y2 = max(x1,0),max(y1,0),min(x2,w),min(y2,h)
        cv2.rectangle(imageBox,(x1,y1),(x2,y2),(0,0,255),2)
        
        cv2.putText(imageBox,str(labels_dict[str(output_dict['detection_classes'][i])]) + ': ' + str(int(output_dict['detection_scores'][i]*100))+'%',
                    (x1, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(imageBox,str(labels_dict[str(output_dict['detection_classes'][i])]) + ': ' + str(int(output_dict['detection_scores'][i]*100))+'%',
                    (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return imageBox

def load_labels(lite = False):
    if lite:
        PATH_TO_LABELS = r"labelmap.txt"
    else:
        PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
    

    with open(PATH_TO_LABELS, 'r') as fh:
        graph_str = fh.read()
        
    if lite:
        Labelsf=graph_str.split('\n')
    else:
        Labelsf=graph_str.split('item')
        
        
    MyLabels = {}
    if lite:
        for n,i in enumerate(Labelsf):
            MyLabels[str(n)] = i
        
    else:
        for index in Labelsf:
            pos1id=index.find('id: ')+4
            pos2id=pos1id+2
            if not index[pos1id:pos2id].isdigit():
                pos2id-=1
            pos1dn=index.find('display_name: ')+15
            pos2dn=index.find('"',pos1dn)
            MyLabels[index[pos1id:pos2id]] = index[pos1dn:pos2dn]
        
    return MyLabels

#Tomar salida de red y dibujar objetos detectados
def detect_all_EfficientDet(image, output_dict, labels_dict, thresh = 0.50):
    imageBox = image.copy()
    for i in output_dict['detections:0'][0]:
        if i[5] > thresh:
            y1,x1,y2,x2 = np.int32(i[1:5].numpy())
            color = (0,255,0)
            if i[6].numpy()==1:
                color = (0,0,255)
            cv2.rectangle(imageBox,(x1,y1),(x2,y2),color,2)
            cv2.putText(imageBox,str(labels_dict[str(int(i[6].numpy()))]) + ': ' + str(int(i[5].numpy()*100))+'%',
                            (x2-50, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return imageBox


#Correr red sobre una imagen
def run_inference_for_single_image_EfficientDet(model, image, automl = False):
    image = np.asarray(image)
    
    if automl:
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]
        outputdictPRE = model(input_tensor)    
        outputdict = {}
        outputdict['detection_boxes'] = outputdictPRE['detections:0'][0][:,1:5].numpy()
        outputdict['detection_boxes'][:,0] = outputdict['detection_boxes'][:,0]/image.shape[0]
        outputdict['detection_boxes'][:,1] = outputdict['detection_boxes'][:,1]/image.shape[1]
        outputdict['detection_boxes'][:,2] = outputdict['detection_boxes'][:,2]/image.shape[0]
        outputdict['detection_boxes'][:,3] = outputdict['detection_boxes'][:,3]/image.shape[1]
        outputdict['detection_classes'] = np.int8(outputdictPRE['detections:0'][0][:,6].numpy())
        outputdict['detection_scores'] = outputdictPRE['detections:0'][0][:,5].numpy()
    else:
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]
        outputdict = model(input_tensor)
        outputdict['detection_boxes'] = outputdict['detection_boxes'][0]
        
        outputdict['detection_classes'] = np.int8(outputdict['detection_classes'][0])
        outputdict['detection_scores'] = outputdict['detection_scores'][0]
        
    return outputdict