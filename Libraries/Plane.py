import cv2
from cv2 import aruco as aruco
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import glob
import pandas as pd
import open3d
import scipy
import scipy.linalg
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
import math
import os
from sklearn import linear_model
import Corners
import pdb
from pdb import set_trace as Debug
from scipy.ndimage import gaussian_filter1d
import time
from sklearn.cluster import DBSCAN
import pcl 

timestart = time.perf_counter()

def tic():
    global timestart
    timestart = time.perf_counter()
    
def toc():
    global timestart
    return time.perf_counter()-timestart

def newnormalaxis(points,normal1,normal2,normal3):
    matrix = np.vstack([normal1,normal2,normal3])
    newpoints = np.matmul(matrix,points.T)
    return newpoints.T

def findconicsection(points):
    ecuation = []
    for i in points:
        ecuation.append([i[0]**2,i[0]*i[1],i[1]**2,i[0],i[1],1])
    ecuation = np.array(ecuation)
    VT=np.linalg.eig(np.matmul(ecuation.T,ecuation))
    eigval = VT[0]
    sort = sorted(zip(eigval,VT[1].T.tolist()),reverse=True)
    _,VT = zip(*sort)
    MF = VT[-1]
    return MF

def conic_center(a):
    a,b,c,d,e,f = a[0], a[1]/2, a[2], a[3]/2, a[4]/2, a[5]
    num = a*c-b*b
    x0=(b*e-c*d)/num
    y0=(b*d-a*e)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])
            
def make2d(points,thresh = 0.04):
    goodpoints,_=Corners.FitPlanePoints(points,thresh=thresh)
    centroid = np.sum(goodpoints,0)/len(goodpoints)
    cgoodpoints = goodpoints-centroid
    VT=np.linalg.eig(np.matmul(cgoodpoints.T,cgoodpoints))
    eigval = VT[0]
    sort = sorted(zip(eigval,VT[1].T.tolist()),reverse=True)
    _,VT = zip(*sort)
    newaxisCenteredchoisepoints=newnormalaxis(cgoodpoints,VT[0],VT[1],VT[2])
    return newaxisCenteredchoisepoints

def make2dnoRAN(goodpoints):
    centroid = np.sum(goodpoints,0)/len(goodpoints)
    cgoodpoints = goodpoints-centroid
    VT=np.linalg.eig(np.matmul(cgoodpoints.T,cgoodpoints))
    eigval = VT[0]
    sort = sorted(zip(eigval,VT[1].T.tolist()),reverse=True)
    _,VT = zip(*sort)
    newaxisCenteredchoisepoints=newnormalaxis(cgoodpoints,VT[0],VT[1],VT[2])
    return newaxisCenteredchoisepoints

def testconicpoint(point,coef):
    i = point
    ecuation=[i[0]**2,i[0]*i[1],i[1]**2,i[0],i[1],1]
    res = np.matmul(ecuation,coef)
    return res

def findclosest(point,points):
    lengths = []
    index = np.arange(len(points))
    for i in points:
        lengths.append(Corners.point_dis(point,i))
    out = [x for _,x in sorted(zip(lengths,index))]
    return out

def findPlanePointswithcentroid(coefs,centroid,points):
    out = []
    newcoefs=[coefs[0],coefs[1],coefs[2],0]
    newpoints = np.subtract(points,centroid)
    for i in newpoints:
        out.append(Corners.Dpoint2Plane(i,newcoefs))
    return out

def makepointvector(coefs,centroid=[0,0,0],length = 1,dense = 10):
    assert len(coefs)==3,'Need x,y,z normalvector'
    newcoefs = np.array(coefs)/np.linalg.norm(np.array(coefs))
    pointline = np.arange(length/dense,length+length/dense,length/dense)
    pointline = np.vstack([pointline,pointline,pointline])
    out = np.add(np.multiply(pointline.T,newcoefs),centroid)
    return out   

def getPrincipalComponents(A):
    centroid = np.mean(A,0)
    A=np.subtract(A,centroid)
    VT=np.linalg.eig(np.matmul(A.T,A))
    eigval = VT[0]
    sort = sorted(zip(eigval,VT[1].T.tolist()),reverse=True)
    Val,VT = zip(*sort)
    return Val,VT,centroid 

def findconictype(coefs,voc=True):
    a,b,c,d,e,f = coefs
    re = b*b-4*a*c
    if re<0:
        if a==c and b==0:
            if voc:
                print('circle')
            nu = 0
        else:
            if voc:
                print('ellipse')
            nu = 1
    elif re == 0:
        if voc:
            print('parabola')
        nu = 2
    else:
        if a+c != 0:
            if voc:
                print('hyperbola')
            nu = 3
        else:
            if voc:
                print('rectangular hyperbola')
            nu = 4
            
            
def FindArcPlanes(inputpoints ,samplesize = 0.02, goodthresh = 0.9, overlap = 3 ,disthesh = 0.03 ):
    if samplesize<1:
        s = int(len(inputpoints)*samplesize)
    else:
        s = samplesize
        
    newstart = int(s/overlap)
    n = 0
    savedlines = []
    savedcentroids = []
    savedpoints = []
    savedlength = []
    savednormals = []
    foundline = False
    i = s
    first = 0
    prethreshold = goodthresh
    while i < len(inputpoints):
        choisepoints = inputpoints[ first : i ]
        flatpoints = Plane.make2dnoRAN(choisepoints)
        coefs = Plane.findconicsection(flatpoints)
        buff = [[i2,k] for i2,k in zip(flatpoints,choisepoints) if abs(Plane.testconicpoint(i2,coefs))<disthesh]
        if len(buff)/len(choisepoints)>prethreshold:  #good found
            #goodpointsflat =[]
            #goodopoints =[]
            #for i3 in buff:
            #    goodpointsflat.append(i3[0])
            #    goodopoints.append(i3[1])
            goodpointsflat,goodopoints = zip(*buff)
            savedpoints = goodopoints
            savedlines.append(savedpoints)
            savecentroid = np.sum(savedpoints,0)/len(savedpoints)
            savedcentroids.append(savecentroid)
            A=goodopoints-savecentroid
            VT=np.linalg.eig(np.matmul(A.T,A))
            eigval = VT[0]
            sort = sorted(zip(eigval,VT[1].T.tolist()),reverse=True)
            _,VT = zip(*sort)
            savednormals.append([VT[0],VT[2]])
            savedlength.append(np.linalg.norm(savedpoints[-1]-savedpoints[0]))
        n += 1
        first = newstart * n
        i = s + first
    return np.array(savedlines),np.array(savedcentroids),np.array(savedlength),np.array(savednormals)


def GetGradient(x):
    [X,Y,Z] = np.split(x,3,1)
    X = X.squeeze()
    Y = Y.squeeze()
    Z = Z.squeeze()
    derivativeX = np.gradient(X)
    derivativeY = np.gradient(Y)
    derivativeZ = np.gradient(Z)
    derivative = np.vstack([derivativeX,derivativeY,derivativeZ]).T
    return derivative

def gaussian(x, sig):
    return 1./(sig*np.sqrt(2.*np.pi))*np.exp(-np.power(x/sig, 2.)/2)

def GaussinaKernel1D(size = 5 ,sigma = 1,step = 1):
    side = size//2
    val = gaussian(np.arange(-side,side+1,step),sigma)
    return val/np.sum(val)

def Der2gaussian(x, sig):
    return -(sig**2-x**2)/((sig**5)*np.sqrt(2.*np.pi))*np.exp(-np.power(x/sig, 2.)/2)

def Der2GaussinaKernel1D(size = 5 ,sigma = 1,step = 1):
    side = size//2
    val = Der2gaussian(np.arange(-side,side+1,step),sigma)
    return val

def bilateral(linepoints,bound = 9,sigInt = 0.03):
    side = bound//2
    matvector = makesignalmatrix(linepoints,bound)
    gausserInt = lambda t: gaussian(t,sigInt)
    funcGausInt = np.vectorize(gausserInt)
    IntensityPart = np.abs(np.subtract(matvector,linepoints))
    Intensitykernel = funcGausInt(IntensityPart)
    W = np.sum(Intensitykernel,axis=0)
    Upper = np.sum(Intensitykernel*matvector,axis = 0)
    return Upper/W

def makesignalmatrix(signal,bound,stride = 1):
    assert bound%2 == 1, 'bound must be odd'
    side = bound//2
    padded = np.concatenate([np.zeros([side]),signal,np.zeros([side])])
    mat = [np.arange(i,len(signal)+i,stride) for i in range(bound)]
    return padded[np.array(mat)]

def Filter3dline(points,sigma = 2):
    X = gaussian_filter1d(points[:,0], sigma)
    Y = gaussian_filter1d(points[:,1], sigma)
    Z = gaussian_filter1d(points[:,2], sigma)
    linepointsFiltered = np.vstack([X,Y,Z]).T
    return linepointsFiltered

def bilateral3dline(points,bound,sigma = 2):
    X = bilateral(points[:,0],bound, sigma)
    Y = bilateral(points[:,1],bound, sigma)
    Z = bilateral(points[:,2],bound, sigma)
    linepointsFiltered = np.vstack([X,Y,Z]).T
    return linepointsFiltered

def bilateral3d(linepoints,bound = 9,sigInt = 0.03):
    side = bound//2
    out = []

    gausserInt = lambda t: gaussian(t,sigInt)
    funcGausInt = np.vectorize(gausserInt)
    matvector = makesignalmatrix(linepoints,bound)
    
    IntensityPart = np.linalg.norm(matvector-linepoints,axis=1)
    Intensitykernel = funcGausInt(IntensityPart)
    W = np.sum(Intensitykernel,axis=0)
    Upper = np.sum(Intensitykernel*matvector,axis = 0)
    
    for n,i in enumerate(linepoints):
        position = n + side
        positionpoints = padded[position-side:position+side+1]
        IntensityPart = np.linalg.norm(positionpoints-padded[position],axis=1)
        Intensitykernel = funcGausInt(IntensityPart)
        buf = Intensitykernel
        W = np.sum(buf)
        Upper = np.sum(buf*positionpoints.T,axis=1)
        out.append(Upper/W)
    return np.array(out)

def bilateral3dNew(linepoints,bound = 9,sigInt = 0.03):
    side = bound//2
    matvector = makesignalmatrix3D(linepoints,bound)
    IntensityPart = np.linalg.norm(np.stack([matvector[0,:,:]-linepoints[0,:],matvector[1,:,:]-linepoints[1,:],matvector[2,:,:]-linepoints[2,:]]), axis = 0)
    Intensitykernel = gaussian(IntensityPart,sigInt)
    W = np.sum(Intensitykernel,axis=0)
    Upper = np.sum(Intensitykernel*matvector,axis = 1)
    return (Upper/W).T


def makesignalmatrix3D(signal,bound,stride = 1):
    assert bound%2 == 1, 'bound must be odd'
    side = bound//2
    dim = min(signal.shape)
    dimB = max(signal.shape)
    padded = np.concatenate([np.zeros([dim,side]),signal,np.zeros([dim,side])],axis = 1)
    mat = [np.arange(i,dimB+i,stride) for i in range(bound)]
    return padded[:,mat]


def GetGradientMulti3(linepoints,span):
    padded = np.concatenate([np.zeros([span,3]),linepoints,np.zeros([span,3])])
    kernel = np.concatenate([-np.arange(1,span+1,1),np.array([0]),np.arange(span,0,-1)])
    derivative = []
    for i in range(len(linepoints)):
        derivative.append(np.matmul(kernel,padded[i:i+span*2+1]))
    return derivative

def Getdirdif3(linepoints,span):
    padded = np.concatenate([np.zeros([span,3]),linepoints,np.zeros([span,3])])
    #kernel = np.concatenate([np.ones(span),[span*-2],np.ones(span)])
    kernel = np.concatenate([np.arange(1,span+1)*0.2+0.8,[-np.sum(np.arange(1,span+1)*0.2+0.8)*2],np.arange(span,0,-1)*0.2+0.8])
    derivative = []
    for i in range(len(linepoints)):
        derivative.append(np.matmul(kernel,padded[i:i+span*2+1]))
    return derivative

def findcurves(points,threshold = 0.005, minimumpoints = 16, sigma = 1, useGradient = False,
               usemul = True, mulspan1 = 2, mulspan2 = 2): 
    #Debug()
    if sigma > 1:
        linepointsFiltered = Filter3dline(points,sigma)
    else:
        linepointsFiltered = bilateral3dNew(points.T,mulspan1,sigma)
    if useGradient:
        derivative1 = GetGradient(linepointsFiltered)
        derivative2 = GetGradient(derivative1)
    elif usemul:
        derivative1 = GetGradientMulti3(linepointsFiltered,mulspan1)
        derivative2 = GetGradientMulti3(derivative1,mulspan2)
    else:
        derivative2 = Getdirdif3(linepointsFiltered,mulspan2)
    segments = []
    segmentbuffer = []
    segmentAccumilator = []
    
    for n,i in enumerate(linepointsFiltered.tolist()):
        if ((np.linalg.norm(derivative2[n])/np.linalg.norm(linepointsFiltered[n])) > threshold):
        #if ((np.linalg.norm(derivative2[n])/np.linalg.norm(linepointsFiltered[n])) > threshold) | len(segmentbuffer) > 30:
            segments.append(i)
            if len(segmentbuffer) > minimumpoints:
                segmentAccumilator.append(segmentbuffer)
            segmentbuffer = []
        else:
            segmentbuffer.append(i)
    return segmentAccumilator

def groupsimilar(curves, distance = 0.05, angle = 20):
    PrinicpalVectors = []
    startpoint = []
    endpoint = []
    newgroups = []
    angleRAD = np.deg2rad(angle)
    breakpoint = True
    for n,i in enumerate(curves):
        savecentroid = np.sum(i,0)/len(i)
        A=i-savecentroid
        VT=np.linalg.eig(np.matmul(A.T,A))
        eigval = VT[0]
        sort = sorted(zip(eigval,VT[1].T.tolist()),reverse=True)
        _,VT = zip(*sort)
        PrinicpalVectors.append(VT)
        startpoint.append(i[0])
        endpoint.append(i[-1])
    
    newgroups.append(curves[0])
    for n,i in enumerate(curves[:-1]):
        pointdistance = np.linalg.norm(np.subtract(endpoint[n],startpoint[n+1]))
        pointangle = Corners.angle_b_vectors(PrinicpalVectors[n],PrinicpalVectors[n+1])
        if (pointdistance < distance) and ( pointangle<angleRAD or pointangle>np.pi-angleRAD ):
            newgroups[-1] = newgroups[-1] + curves[n+1]
        else:
            newgroups.append(curves[n+1])
    return newgroups


def Beamplaneacumulate(BeamDF, threshold = 0.05, minimumpoints = 16, distance = 0.1, angle = 20, sigma = 1, usemul = False,
                       useGradient = False, mulspan1 = 2, mulspan2 = 3 ):
    BeamAcumulator = []
    lasersequence=[15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0]
    for i in lasersequence:
        linepointsDF=BeamDF[(BeamDF['laser_id']==(i))]
        #linepoints=linepointsDF.sort_values(by=['azimuth']).iloc[:,1:4].values
        linepoints=linepointsDF[['X','Y','Z']].values
        #segmentAccumilator = findcurves(linepoints,threshold = 0.001,minimumpoints = 12)
        #segmentAccumilator = findcurves(linepoints,threshold = 0.05,minimumpoints = 16,
        #                              sigma = 1,usemul = True,mulspan1 = 3, mulspan2 = 3 )  
        segmentAccumilator = findcurves(linepoints,threshold = threshold, minimumpoints = minimumpoints,
                                        sigma = sigma, usemul = usemul, useGradient = useGradient,
                                        mulspan1 = mulspan1, mulspan2 =mulspan2)
        if distance > 0.00001:
            segmentAccumilator = groupsimilar(segmentAccumilator, distance = distance, angle = angle)
        BeamAcumulator.append(segmentAccumilator)
    return BeamAcumulator

def SmoothBeams(BeamDF, mulspan1 = 9, sigma = 0.04):
    BeamAcumulator = []
    lasersequence=[15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0]
    for i in lasersequence:
        linepointsDF=BeamDF[(BeamDF['laser_id']==(i))]
        linepoints=linepointsDF[['X','Y','Z']].values
        segmentAccumilator = bilateral3dNew(linepoints.T,mulspan1,sigma)
        BeamAcumulator.append(segmentAccumilator)
    return BeamAcumulator

    
def findplaneproposal2(initial,centroids,curves,PVs):
    curvenumber = initial
    closestpointsid = findclosest(np.array(centroids[curvenumber]),centroids)
    n = 1
    success = False
    options = []
    votes = []
    while n < 10:
        closestpoints = curves[closestpointsid[0]] + curves[closestpointsid[n]]
        val,vectorspace,vectorcentroid = getPrincipalComponents(closestpoints)
        normalvector = homovector(vectorspace[2])
        pointangle = Corners.angle_b_vectors(PVs[curvenumber][0],PVs[closestpointsid[n]][0])
        distance = np.linalg.norm(centroids[closestpointsid[0]]-centroids[closestpointsid[n]])
        if (distance > (0.4*np.linalg.norm(centroids[closestpointsid[0]]))):
            break
            
        if (val[0]/val[1]<80) & (pointangle<0.34906 or pointangle>np.pi-0.34906) & (val[1]/val[2]>200):
            found = False
            for nn,i in enumerate(options):
                angg = Corners.angle_b_vectors(normalvector,i)
                if (pointangle<0.17 or pointangle>np.pi-0.17):
                    votes[nn] += 1
                    found = True
            if found == False:
                options.append(normalvector.tolist())
                votes.append(1)
        n += 1
    
    if len(votes)>0:
        sort = sorted(zip(votes,options),reverse=True)
        votes,options = zip(*sort)
        votes = list(votes)
        options = list(options)
    
    return options,votes


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def homovector(vector):
    assert len(vector) == 3,'vector must be dim 3' 
    assert (vector[0] != 0) | (vector[1] != 0) | (vector[2] != 0),'vector is trivial' 
    s1 = np.sign(vector[0])
    s2 = np.sign(vector[1])
    s3 = np.sign(vector[2])

    if s1 != 0:
        vector = np.multiply(vector,s1)
    elif s2 != 0:
        vector = np.multiply(vector,s2)
    elif s3 != 0:
        vector = np.multiply(vector,s3)
    return vector

def similarize(vector,target):
    Nvector = np.array(vector)
    assert len(Nvector) == 3,'vector must be dim 3'
    angle = Corners.angle_b_vectors(Nvector,target)
    if angle > np.pi/2:
        Nvector = -Nvector
    return Nvector


def preparedata(acu):
    AllCurves = [o for i in acu for o in i]
    Allcentroids = []
    AllPV = []

    for i in AllCurves:
        val,vec,cen = getPrincipalComponents(i)
        Allcentroids.append(cen)
        AllPV.append(vec)
        
    return AllCurves, Allcentroids, AllPV

def findplane(points,thresh=0.05):
    p = np.float32(points)
    cloud = pcl.PointCloud_PointXYZRGB()
    cloud.from_array(np.hstack([p,np.ones([len(p),1],dtype = np.float32)]))
    seg = cloud.make_segmenter()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(thresh)
    inliers, model = seg.segment()
    mask = np.isin(np.arange(p.shape[0]), inliers)
    return p[mask],mask,model


def planefinderHough2(BoundDF):
    verbosity = True
    #acu = Beamplaneacumulate(BoundDF,threshold = 0.14, minimumpoints = 14, distance = 0.001, angle = 0.5, sigma = 5, usemul = False, mulspan1 = 2, mulspan2 = 8 ,useGradient = False)

    acu = Beamplaneacumulate(BoundDF,threshold = 0.2, minimumpoints = 12,
                                      distance = 0.1, angle = 5, sigma = 0.08, usemul = False, 
                                      mulspan1 = 21, mulspan2 = 21 ,useGradient = False)
    AllCurves, Allcentroids, AllPV = preparedata(acu)   
    worklines=np.array(AllCurves)
    workcentroids=np.array(Allcentroids)
    workPVs=np.array(AllPV)
    listoptions = []
    listvotes = []
    listlines = []
    listcenters = []
    for n,i in enumerate(worklines):
        options,votes = findplaneproposal2(n,workcentroids,worklines,workPVs)
        if len(votes)>0:
            listlines.append(worklines[n])
            listoptions.append(options[0])
            listvotes.append(votes)
            listcenters.append(workcentroids[n])
            
    maxlen = max([len(i) for i in listvotes])
    buff = [list(i) + np.zeros(maxlen-len(i)).tolist() for i in listvotes]
    probs = scipy.special.softmax(buff,1)
    goodcurves = (np.max(probs,1) > 0.6)
            
    planesslot = np.array(listlines)[goodcurves].tolist()
    goodnormals = np.array(listoptions)[goodcurves].tolist()
    goodcenters = np.array(listcenters)[goodcurves].tolist()

    planes = []
    centers = []
    
    sphericals = np.array([cart2sph(i[0], i[1], i[2]) for i in goodnormals])
    
    db2 = DBSCAN(eps = 0.1, min_samples=5).fit(sphericals[:,0:2])
    for i in np.unique(db2.labels_):
        if i == -1:
            continue
        pp = np.array(planesslot)[db2.labels_==i].tolist()
        centers.append(np.array(goodcenters)[db2.labels_==i])
        planes.append(pp)
    finalplanes = []
    finalcenters = []
    
    for i in range(len(planes)):
        points = [o for i2 in planes[i] for o in i2]
        left = len(points)
        buffer = np.array(points)
        while left>len(points)*0.05:
            
            plane,mask,_ = findplane(buffer,0.06)
            if len(plane)<200:
                break
            finalplanes.append(plane)
            left = len(buffer[~mask])
            buffer = buffer[~mask]
    
    return finalplanes
    #return finalplanes,planes,planesslot,goodnormals,goodcenters
    
    
def planefinderHough2diag(BoundDF):
    verbosity = True
    #acu = Beamplaneacumulate(BoundDF,threshold = 0.14, minimumpoints = 14, distance = 0.001, angle = 0.5, sigma = 5, usemul = False, mulspan1 = 2, mulspan2 = 8 ,useGradient = False)

    acu = Beamplaneacumulate(BoundDF,threshold = 0.2, minimumpoints = 12,
                                      distance = 0.000002, angle = 1, sigma = 0.08, usemul = False, 
                                      mulspan1 = 21, mulspan2 = 21 ,useGradient = False)
    AllCurves, Allcentroids, AllPV = preparedata(acu)   
    worklines=np.array(AllCurves)
    workcentroids=np.array(Allcentroids)
    workPVs=np.array(AllPV)
    listoptions = []
    listvotes = []
    listlines = []
    listcenters = []
    for n,i in enumerate(worklines):
        options,votes = findplaneproposal2(n,workcentroids,worklines,workPVs)
        if len(votes)>0:
            listlines.append(worklines[n])
            listoptions.append(options[0])
            listvotes.append(votes)
            listcenters.append(workcentroids[n])
            
    maxlen = max([len(i) for i in listvotes])
    buff = [list(i) + np.zeros(maxlen-len(i)).tolist() for i in listvotes]
    probs = scipy.special.softmax(buff,1)
    goodcurves = (np.max(probs,1) > 0.6)
            
    planesslot = np.array(listlines)[goodcurves].tolist()
    goodnormals = np.array(listoptions)[goodcurves].tolist()
    goodcenters = np.array(listcenters)[goodcurves].tolist()

    planes = []
    centers = []
    
    sphericals = np.array([cart2sph(i[0], i[1], i[2]) for i in goodnormals])
    
    db2 = DBSCAN(eps = 0.1, min_samples=5).fit(sphericals[:,0:2])
    for i in np.unique(db2.labels_):
        if i == -1:
            continue
        pp = np.array(planesslot)[db2.labels_==i].tolist()
        centers.append(np.array(goodcenters)[db2.labels_==i])
        planes.append(pp)
    finalplanes = []
    finalcenters = []
    
    for i in range(len(planes)):
        points = [o for i2 in planes[i] for o in i2]
        left = len(points)
        buffer = np.array(points)
        while left>len(points)*0.05:
            
            plane,mask,_ = findplane(buffer,0.06)
            if len(plane)<200:
                break
            finalplanes.append(plane)
            left = len(buffer[~mask])
            buffer = buffer[~mask]
    
    return finalplanes,planes,planesslot,goodnormals,goodcenters
    
def makesphere(centroid=[0,0,0],radius = 1,dense = 90):
    n = np.arange(0,360,int(360/dense))
    n = np.deg2rad(n)
    x,y = np.meshgrid(n,n)
    x = x.flatten()
    y = y.flatten()
    sphere = np.vstack([centroid[0]+np.sin(x)*np.cos(y)*radius,centroid[1]+np.sin(x)*np.sin(y)*radius,centroid[2]+np.cos(x)*radius]).T
    return sphere      
    
"""    
class ArucoCam:
    def __init__(self,hasIntrinsic=True, file=None):
        if hasIntrinsic:
            with np.load(file+'.npz') as L:
                self.camera_matrix, self.dist_coefs, self.new_camera_matrix, self.seedH, self.seedW =\
                [L[i] for i in ('camera_matrix','dist_coefs','new_camera_matrix','seedH','seedW')]
            
    def findCharucos(frame,arucodict):
        arucodict1=aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        arucodict2=aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        arucodict3=aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        markersize = 0.0848
        squaresize = 0.1062
        #squaresize = 0.08482
        charboard1=aruco.CharucoBoard_create(9,7,squaresize,markersize,arucodict1) 
        charboard2=aruco.CharucoBoard_create(9,7,squaresize,markersize,arucodict2) 
        charboard3=aruco.CharucoBoard_create(9,7,squaresize,markersize,arucodict3) 
        paramiters=cv2.aruco.DetectorParameters_create()
        paramiters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        paramiters.adaptiveThreshWinSizeMax = 33
        paramiters.adaptiveThreshWinSizeStep = 16

        rot1=[]
        listids1=[]
        tra1=[]
        rot2=[]
        listids2=[]
        tra2=[]
        rot3=[]
        listids3=[]
        tra3=[]

        findingaruco1,ids1,reg1=aruco.detectMarkers(frame,arucodict1,parameters =paramiters)
        if len(findingaruco1)>0:
            findingaruco2,ids2,reg2=aruco.detectMarkers(frame,arucodict2,parameters =paramiters)
            if len(findingaruco2)>0:
                findingaruco3,ids3,reg3=aruco.detectMarkers(frame,arucodict3,parameters =paramiters)
                if len(findingaruco3)>0:            
                    rvec1,tvec1,objectpoints1=aruco.estimatePoseSingleMarkers(findingaruco1,markersize,camera_matrix,dist_coefs)
                    rvec2,tvec2,objectpoints2=aruco.estimatePoseSingleMarkers(findingaruco2,markersize,camera_matrix,dist_coefs)
                    rvec3,tvec3,objectpoints3=aruco.estimatePoseSingleMarkers(findingaruco3,markersize,camera_matrix,dist_coefs)
                    rot1.append(rvec1[ids1.argsort(0)].squeeze())
                    tra1.append(tvec1[ids1.argsort(0)].squeeze())
                    listids1.append(ids1[ids1.argsort(0)].squeeze())
                    rot2.append(rvec2[ids2.argsort(0)].squeeze())
                    tra2.append(tvec2[ids2.argsort(0)].squeeze())
                    listids2.append(ids2[ids2.argsort(0)].squeeze())
                    rot3.append(rvec3[ids3.argsort(0)].squeeze())
                    tra3.append(tvec3[ids3.argsort(0)].squeeze())
                    listids3.append(ids3[ids3.argsort(0)].squeeze())
        IntCor1 = cv2.aruco.interpolateCornersCharuco(findingaruco1,ids1,frame,charboard1,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
        _,Brvec1,Btvec1=cv2.aruco.estimatePoseCharucoBoard(IntCor1[1], IntCor1[2], charboard1, camera_matrix, dist_coefs)

        IntCor2 = cv2.aruco.interpolateCornersCharuco(findingaruco2,ids2,frame,charboard2,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
        _,Brvec2,Btvec2=cv2.aruco.estimatePoseCharucoBoard(IntCor2[1], IntCor2[2], charboard2, camera_matrix, dist_coefs)

        IntCor3 = cv2.aruco.interpolateCornersCharuco(findingaruco3,ids3,frame,charboard3,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
        _,Brvec3,Btvec3=cv2.aruco.estimatePoseCharucoBoard(IntCor3[1], IntCor3[2], charboard3, camera_matrix, dist_coefs)

        return Brvec1,Btvec1,Brvec2,Btvec2,Brvec3,Btvec3
    
    def findMarkersCharuco(image = None,file = None , chardict = 4, , equalize = 0):
        assert 4<=chardict<=6,'Chardict must be between 4 and 6'
        assert (image is not None) != (file is not None),'Need input image'
        
        allCorners  = [] # 3d point in real world space
        allIds  = []
        allDirs  = []
        
        if chardict ==4:
            self.arucodict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        elif chardict ==6:
            self.arucodict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        elif chardict ==5:
            self.arucodict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        
        paramiters=cv2.aruco.DetectorParameters_create()
        paramiters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        paramiters.adaptiveThreshWinSizeMax = 103
        paramiters.adaptiveThreshWinSizeStep = 82
        
        if image is not None:
            imgc = image
            
        if image is not None:
            imgc = cv2.imread(fname)
            
        imgc = cv2.resize(imgc,(seedW,seedH))
        
        if equalize != 0:
            imgc = cv2.min(imgc,150)
            imgc = cv2.equalizeHist(imgc)
            
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(imgc, arucodict,parameters =paramiters)
        if hasIntrinsic:
            res2 = cv2.aruco.interpolateCornersCharuco(findingaruco1,ids1,frame,arucodict,cameraMatrix =self.camera_matrix,distCoeffs =self.dist_coefs)
        else:
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,imgc,charboard1)
        self.corners = corners  
        self.ids = ids
        self.charcorners = res2[1]
        self.charids = res2[2]
        return corners,ids,res2
    
    def getcharucoPose():
    _,Brvec1,Btvec1=cv2.aruco.estimatePoseCharucoBoard(self.charcorners, self.charids, self.arucodict, self.camera_matrix, self.dist_coefs)
        rvec1,tvec1,objectpoints1=aruco.estimatePoseSingleMarkers(findingaruco1,markersize,camera_matrix,dist_coefs) 
"""