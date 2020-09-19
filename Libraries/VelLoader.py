import time
import numpy as np
import pandas as pd
from Plane import tic,toc
from pypacker import ppcap
from pypacker.layer12 import ethernet
from pypacker.layer3 import ip
from pypacker.layer4 import tcp

class VVdataloader:
    angles = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
    verticalcorrection = [0.0112, -0.0007, 0.0097, -0.0022, 0.0081, -0.0037, 0.0066, -0.0051, 0.0051, -0.0066, 0.0037, -0.0081, 0.0022, -0.0097, 0.007, -0.00112]
    
    def __init__(self, file):
        #self.capfile = savefile.load_savefile(open(file, 'rb'), verbose=True)
        self.capfile = ppcap.Reader(filename=file)
        #self.packets = [i.raw() for i in self.capfile.packets]
        self.packets = [] 
        self.packets = [buf for ts, buf in self.capfile]
        self.packets = [i for i in self.packets if i[42:44] == b'\xff\xee']
        self.dualreturn = int.from_bytes(self.packets[0][1246:1247], byteorder='big') == 57
        self.packettimestamplist = self.listoftimestamps(self.packets)
        self.timing_offsets = self.make_table(self.dualreturn)
        self.blockazimuths = self.azimuthblocklist()
        self.flatazimuths = [n for i in self.blockazimuths for n in i]
        self.frames = self.findframes()
        #pd.set_option('float_format', '{:f}'.format)
    
    def getMultiFrame(self, start, NF):
        n = start
        for i in range(NF):
            n += i 
            if i == 0:
                LIDARDFpoints = self.getdataFrame(n)
            else:
                LIDARDFpoints = pd.concat([LIDARDFpoints,self.getdataFrame(n)])
        LIDARDFpoints = LIDARDFpoints.sort_values('Azimuth')
        LIDARDFpoints = LIDARDFpoints.reset_index()
        LIDARpoints = LIDARDFpoints[['X','Y','Z']].values
            
        return LIDARDFpoints,LIDARpoints
    
    def getdataFrame(self,frame):
        start = [0,0]
        oneshot = True
        if frame > 0:
            start =[self.frames[frame-1][0],self.frames[frame-1][1]]
        end =[self.frames[frame][0]+1,self.frames[frame][1]]
        
        points = []
        for i in range(start[0],end[0]):            
            for blocknumber in range(12):
                firstazimuth = self.flatazimuths[i*12+blocknumber]
                if firstazimuth != self.flatazimuths[i*12+blocknumber+1]:
                    secondazimuth = self.flatazimuths[i*12+blocknumber+1]
                else:
                    secondazimuth = self.flatazimuths[i*12+blocknumber+2]
                if (secondazimuth < firstazimuth):
                    AzimuthGap = secondazimuth - firstazimuth + 360
                else:
                    AzimuthGap = secondazimuth - firstazimuth
                
                index = 4
                bytepart = self.packets[i][42+blocknumber*100:142+blocknumber*100]
                for m in range(32):
                    
                    if (m < 16):
                        Precision_Azimuth = firstazimuth + (AzimuthGap * 2.304 * m) / 55.296; 
                    else:
                        Precision_Azimuth = firstazimuth + (AzimuthGap * 2.304 * ((m-16) + 55.296)) / (2 * 55.296);
                    if Precision_Azimuth >= 360:
                        Precision_Azimuth = Precision_Azimuth - 360
                    distance, reflectance = self.singledata(bytepart[index:index+3])
                    angle = self.angles[m%16]
                    X,Y,Z = self.calcPOS(distance,Precision_Azimuth,angle,m%16)
                    if self.dualreturn and blocknumber%2 == 1:
                        if self.singledata(bytepart[index:index+3])[0] != self.singledata(lastblock[index:index+3])[0]:
                            points.append([distance, reflectance, Precision_Azimuth] + [X,Y,Z] +
                                      [self.packettimestamplist[i] + self.timing_offsets[m][blocknumber]] +
                                         [m%16] + [angle])
                    else:
                        points.append([distance, reflectance, Precision_Azimuth] + [X,Y,Z] +
                                      [self.packettimestamplist[i] + self.timing_offsets[m][blocknumber]] +
                                         [m%16] + [self.angles[m%16]])
                    index += 3
                lastblock = bytepart
        points = list(filter(lambda x: x[0] != 0,points))
        start = next(n for n,x in enumerate(points) if x[2]<10)
        end = len(points) - next(n for n,x in enumerate(reversed(points)) if x[2]>350)
        points = points[start:end]
        pointsDF = pd.DataFrame(points,columns = ['Distance','Intensity','Azimuth','X','Y','Z','TimeStamp','laser_id','Angle'])
        pointsDF = pointsDF.sort_values(by=['TimeStamp'])
        return pointsDF
                
      
    def calcPOS(self,dist,azi,ang,n):
        alfa = np.deg2rad(azi)
        omega = np.deg2rad(ang)
        X = dist*np.cos(omega)*np.sin(alfa)
        Y = dist*np.cos(omega)*np.cos(alfa)
        Z = dist*np.sin(omega) - self.verticalcorrection[n]
        return X,Y,Z
        
    def findframes(self):
        oneshot = True
        framelist = []
        for y in range(len(self.blockazimuths)):
            for x in range(12):
                if oneshot:
                    oneshot = False
                    continue 
                if x == 0:
                    if (self.blockazimuths[y-1][11] > self.blockazimuths[y][0]):
                        framelist.append([y,0])
                else:
                    if (self.blockazimuths[y][x-1] > self.blockazimuths[y][x]):
                        framelist.append([y,x])
        return framelist
    
    def azimuthblocklist(self):
        #packetazimuths = [[self.azimuthcalc(values[44+i2:46+i2]) for i2 in range(0,1200,100)] for values in self.packets]
        packetazimuths = [[int.from_bytes(values[44:46], byteorder='little')/100,
                   int.from_bytes(values[144:146], byteorder='little')/100,
                   int.from_bytes(values[244:246], byteorder='little')/100,
                   int.from_bytes(values[344:346], byteorder='little')/100,
                   int.from_bytes(values[444:446], byteorder='little')/100,
                   int.from_bytes(values[544:546], byteorder='little')/100,
                   int.from_bytes(values[644:646], byteorder='little')/100,
                   int.from_bytes(values[744:746], byteorder='little')/100,
                   int.from_bytes(values[844:846], byteorder='little')/100,
                   int.from_bytes(values[944:946], byteorder='little')/100,
                   int.from_bytes(values[1044:1046], byteorder='little')/100,
                   int.from_bytes(values[1144:1146], byteorder='little')/100
                  ] for values in self.packets]
        return packetazimuths
       
    def listoftimestamps(self,packets):
        simplelist = [self.timestampcalc(x[1242:1246]) for x in packets]
        return simplelist
    
    def distancecalc(self, byte1 ):
        return int.from_bytes(byte1, byteorder='little')*0.002
    
    def azimuthcalc(self, byte1):
        return int.from_bytes(byte1, byteorder='little')/100
    
    def timestampcalc(self, byte1):
        return int.from_bytes(byte1, byteorder='little')
    
    def singledata(self,bytepart):  
        return [self.distancecalc(bytepart[0:2]),bytepart[2]]
    
    def make_table(self,dual_mode): 
        timing_offsets = [[0.0 for x in range(12)] for y in range(32)] # Init matrix 
        # constants 
        full_firing_cycle = 55.296 # μs 
        single_firing = 2.304 # μs 
        # compute timing offsets 
        for x in range(12): 
            for y in range(32): 
                if dual_mode: 
                    dataBlockIndex = (x - (x % 2)) + (y // 16) 
                else: 
                    dataBlockIndex = (x * 2) + (y // 16) 
                dataPointIndex = y % 16 
                timing_offsets[y][x] = (full_firing_cycle * dataBlockIndex) + (single_firing * dataPointIndex)
        return timing_offsets
    
