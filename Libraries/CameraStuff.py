import sys
import cv2
import numpy as np
import os

def showim(image,size=(1820,980)):
    if type(image) is list:
        if len(image[0].shape) == 3:
            h,w,_ = image[0].shape
            newimages = [image[0]]
        else:
            h,w = image[0].shape
            newimages = [GRAY3Channels(image[0])]
            
        for i in image[1:]:
            if len(i.shape) == 3:
                newimages.append(cv2.resize(i,(w,h)))
            else:
                newimages.append(cv2.resize(GRAY3Channels(i),(w,h)))
                
        image = np.hstack(newimages)
        
    if size is not None:
        image = cv2.resize(image.copy(),size)
        
    cv2.imshow('',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def showimKey(image,size=(1820,980),timing = 0):
    k = -1
    if type(image) is list:
        if len(image[0].shape) == 3:
            h,w,_ = image[0].shape
            newimages = [image[0]]
        else:
            h,w = image[0].shape
            newimages = [GRAY3Channels(image[0])]
        
        for i in image[1:]:
            if len(i.shape) == 3:
                newimages.append(cv2.resize(i,(w,h)))
            else:
                newimages.append(cv2.resize(GRAY3Channels(i),(w,h)))
                
        image = np.hstack(newimages)
        
    if size is not None:
        image = cv2.resize(image.copy(),size)
    cv2.imshow('',image)
    k = cv2.waitKey(timing)
    return k
        
    
    
def ROIim(image,size=(1820,980),coordinates=True):
    ratio = [1,1]
    if size is not None:
        ratio = [image.shape[1]/size[0],image.shape[0]/size[1]]
        imageR = cv2.resize(image.copy(),size)
    x = cv2.selectROI(imageR)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    x2 = [int(x[0]*ratio[0]) , int(x[1]*ratio[1]), int(x[2]*ratio[0]), int(x[3]*ratio[1])]
    if coordinates:
        return x2
    else:
        return image.copy()[x2[1]:x2[1]+x2[3],x2[0]:x2[0]+x2[2]]
    
def BGR2GRAY(image):
    assert type(image) is np.ndarray
    assert len(image.shape) == 3 
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def RGBSwap(image):
    assert type(image) is np.ndarray
    assert len(image.shape) == 3
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def GRAY3Channels(image):
    assert type(image) is np.ndarray
    assert len(image.shape) == 2 
    return cv2.merge([image,image,image])


class AutoGui:
    def nothing(self,x):
        pass
    
    def __init__(self,variables = 1,windowname = 'settings_*',initialvalues = 0,maxvalues = 255):
        assert type(windowname) is str, 'windowname needs to be string or contain _* for incrimental'        
        if '_*' in windowname:
            windownumer = 0
            while True:
                if cv2.getWindowProperty(windowname.replace('_*','_') + str(windownumer), cv2.WND_PROP_VISIBLE) == 0:
                    break
                else:
                    windownumer += 1
            self.FinalWindowName = windowname.replace('_*','_') + str(windownumer)
            cv2.namedWindow(self.FinalWindowName)
        else:            
            assert cv2.getWindowProperty(windowname, cv2.WND_PROP_VISIBLE) == 0 , 'A window already exists with that name'
            cv2.namedWindow(windowname)
            
            
        if type(variables) is int:
            self.myvariables = np.arange(variables)
        else:
            assert (type(variables) is list) & ((type(initialvalues) is list) | (type(initialvalues) is int)) & ((type(maxvalues) is list) | (type(maxvalues) is int)),'variables, initialvalues and maxvalues have to be an int or a list'
            self.myvariables = variables
        
        if type(initialvalues) is int:
            self.initialvalues = np.ones(len(self.myvariables), dtype = np.int32)*initialvalues
        else:
            assert type(initialvalues) is list , 'variables must be list or int'
            assert len(self.myvariables) == len(initialvalues) , 'initialvalues and maxvalues must be an int or a list of intsthe same length as variables'
            assert all([type(i) is int for i in initialvalues]), 'initialvalues and maxvalues must be an int or a list of ints the same length as variables'
            self.initialvalues = initialvalues
        
        if type(maxvalues) is int:
            self.maxvalues = np.ones(len(self.myvariables), dtype = np.int32)*maxvalues
        else:
            assert type(maxvalues) is list , 'variables must be list or int'
            assert len(self.myvariables) == len(maxvalues) , 'initialvalues and maxvalues must be an int or a list of intsthe same length as variables'
            assert all([type(i) is int for i in maxvalues]), 'initialvalues and maxvalues must be an int or a list of ints the same length as variables'
            self.maxvalues = maxvalues
        self.makebars()
            
    def makebars(self):
        for i,n,g in zip(self.myvariables,self.initialvalues,self.maxvalues):
            cv2.createTrackbar(str(i),self.FinalWindowName,n,g,self.nothing)

    def getbars(self):
        returns = []
        for i in self.myvariables:
            returns.append(cv2.getTrackbarPos(str(i),self.FinalWindowName))
        return returns

    def close(self):
        cv2.destroyWindow(self.FinalWindowName)