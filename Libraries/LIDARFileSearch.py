import numpy as np
import cv2
import csv
import pandas as pd
import exifread
import glob
from matplotlib import pyplot as plt

def closest_LIDAR_frame(hrs,mins,secs,file='LIDAR_Times.csv'):
    index=1
    searchstamp=min_secto_timestamp(hrs,mins,secs)
    
    raw_dataset = pd.read_csv(file,
                      na_values = "?", comment='\t',
                    sep=",", skipinitialspace=True,header = None, names = ['index','file','times'] )
    startstamp = raw_dataset.at[0,'times']
    
    while index<len(raw_dataset):
        stamp = raw_dataset.at[index,'times']

        index+=1
        if stamp>searchstamp or startstamp>stamp:
            index-=2
            break;
        if index==len(raw_dataset):
            print('not found')
            
        
    return raw_dataset.at[index,'index'],raw_dataset.at[index,'file'],raw_dataset.at[index,'times']

def closest_LIDAR_frame(searchstamp,file='LIDAR_Times.csv'):
    index=1
    
    raw_dataset = pd.read_csv(file,
                      na_values = "?", comment='\t',
                    sep=",", skipinitialspace=True,header = None, names = ['index','file','times'] )
    startstamp = raw_dataset.at[0,'times']
    
    while index<len(raw_dataset):
        stamp = raw_dataset.at[index,'times']

        index+=1
        if stamp>searchstamp or startstamp>stamp:
            index-=2
            break
        if index==len(raw_dataset):
            print('not found')
            break
    return raw_dataset.at[index,'index'],raw_dataset.at[index,'file'],raw_dataset.at[index,'times']

def make_lidar_times(files,outfile='LIDAR_Times'):
    times=[]
    for index in range(len(files)):
        raw_dataset = pd.read_csv(files[index],
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True,nrows=1)
        middlepoint=np.int16(len(raw_dataset)/2)
        times.append([index,files[index],raw_dataset.at[middlepoint,'adjustedtime']])
        if index%1000==0:
            print(index)
    newlist=sorted(times, key=lambda times: times[2])
    return np.savetxt(outfile+'.csv', newlist, delimiter=",",fmt="%s")


def make_Big_LiDAR_File(files,outfile='LIDAR_Times'):
    times=[]
    for index in range(len(files)):
        raw_dataset = pd.read_csv(files[index],
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True,nrows=1)
        middlepoint=np.int16(len(raw_dataset)/2)
        times.append([index,files[index],raw_dataset.at[middlepoint,'adjustedtime']])
        if index%1000==0:
            print(index)
    newlist=sorted(times, key=lambda times: times[2])
    return np.savetxt(outfile+'.csv', newlist, delimiter=",",fmt="%s")

def timestampto_min_sec(time):
    hours=np.trunc(time/(3600*1000000))
    minutes=np.trunc((time-hours*(3600*1000000))/(60*1000000))
    seconds=(time-hours*(3600*1000000)-minutes*(60*1000000))/1000000
    return hours,minutes,seconds


def min_secto_timestamp(hours,minutes, seconds):
    redo=hours*3600*1000000+minutes*60*1000000.+seconds*1000000.
    return redo

def join_LIDAR_times(files1,file2,outfile='LIDAR_Times'):
    raw_dataset1 = pd.read_csv(files1,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True,header = None, names = ['index','file','times'] )
    raw_dataset2 = pd.read_csv(file2,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True,header = None, names = ['index','file','times'] )
    dataset=pd.concat([raw_dataset1,raw_dataset2])
    times=dataset.values
    newlist=sorted(times, key=lambda times: times[2])
    np.savetxt(outfile+'.csv', newlist, delimiter=",",fmt="%s")

def FindTimeLimits(times, searchtime):
    assert(times[0]<searchtime or times[-1]>searchtime), 'search time outside list'
    LeftLim = 0
    RightLim = 1
    for n,i in enumerate(times):
        if i > searchtime:
            RightLim = n
            LeftLim = n-1
            break
    return LeftLim,RightLim

converttime = lambda x: min_secto_timestamp(x.hour-5,x.minute,x.second)

def polate(Dataframe,Num,Time):
    Left = converttime(Dataframe['time'][Num])
    Right = converttime(Dataframe['time'][Num + 1])
    Valx1 = Dataframe['lon'][Num]
    Valx2 = Dataframe['lon'][Num+1]
    
    Valy1 = Dataframe['lat'][Num]
    Valy2 = Dataframe['lat'][Num+1]
    
    Valz1 = Dataframe['alt'][Num]
    Valz2 = Dataframe['alt'][Num+1]
    
    valuex = (Valx2-Valx1)/(Right-Left)*(Time-Left)+Valx1
    valuey = (Valy2-Valy1)/(Right-Left)*(Time-Left)+Valy1
    valuez = (Valz2-Valz1)/(Right-Left)*(Time-Left)+Valz1
    return valuex,valuey,valuez