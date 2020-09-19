import cv2
from cv2 import aruco as aruco
from matplotlib import pyplot as plt
from matplotlib import cm
import pclpy
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
import Plane
from pdb import set_trace as Debug

def point_dis(a,b=None):
    if b is not None:
        c=a-b
    else:
        c=a
    anorm=np.sqrt(np.sum(np.multiply(c,c)))
    return anorm

def angle_b_vectors(a,b):
    value = np.sum(np.multiply(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    if (value<-1) | (value>1):
        value = np.sign(value)
    angle = np.arccos(value)
    return angle

def get_LIDAR(file,names=None,infer=True):
    if infer:
        raw_dataset = pd.read_csv(file,
                      na_values = "?", comment='\t',
                    sep=",", skipinitialspace=True)
    else:
        if names is not None:
            raw_dataset = pd.read_csv(file,
                      na_values = "?", comment='\t',
                    sep=",", skipinitialspace=True,names  = names)
        else:
            raw_dataset = pd.read_csv(file,
                      na_values = "?", comment='\t',
                    sep=",", skipinitialspace=True,header  = None)
    return raw_dataset

def convertcloud(points):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    #open3d.write_point_cloud(Path+'sync.ply', pcd)
    #pcd_load = open3d.read_point_cloud(Path+'sync.ply')
    return pcd

def cloudsave(points,Path):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    open3d.write_point_cloud(Path+'.ply', pcd)
    pcd_load = open3d.read_point_cloud(Path+'.ply')
    return pcd_load

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ]) 
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])     
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def eulerAnglesToRotationMatrixZYX(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ]) 
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])     
    R = np.dot(R_x, np.dot( R_y, R_z ))
    return R

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAnglesZYX(R) :
    assert(isRotationMatrix(R))
     
    if(R[2,0]<1):
        if (R[2,0]>-1):
            y = np.arcsin(-R[2,0]) 
            z = np.arctan2(R[1,0],R[0,0])
            x = np.arctan2(R[2,1],R[1,1])
        else:
            y = np.pi/2
            z = -np.arctan2(-R[1,2],R[1,1])
            x = 0
    else :
        y = -np.pi/2
        z = np.arctan2(-R[1,2],R[1,1])
        x = 0
 
    return np.array([x, y, z])

def rotationMatrixToEulerAnglesYZX(R) :
    assert(isRotationMatrix(R))
     
    if(R[1,0]<1):
        if (R[1,0]>-1):
            z = np.arcsin(R[1,0]) 
            y = np.arctan2(-R[2,0],R[0,0])
            x = np.arctan2(-R[1,2],R[1,1])
        else:
            z = -np.pi/2
            y = -np.arctan2(-R[2,1],R[2,2])
            x = 0
    else :
        z = -np.pi/2
        y = np.arctan2(R[2,1],R[2,2])
        x = 0
 
    return np.array([x, y, z])

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def model(v1,coeffs):
    R=eulerAnglesToRotationMatrix(coeffs)
    RotatedV=np.matmul(R,v1.transpose())
    resultado=RotatedV.transpose()
    return resultado

def hughtransform_1p(point,Itheta,Iphi):
    p=point[0]*np.cos(Itheta)*np.sin(Iphi)+point[1]*np.sin(Itheta)*np.sin(Iphi)+point[2]*np.cos(Iphi)
    return p

def hugh3p_inter(coefs,p1,p2,p3,P):
    v1=hughtransform_1p((p1-p2-p3),coefs[0],coefs[1])-P
    return v1

def LIDAR_Camera_Join(image,camera_matrix,dist_coefs,LIDARpoints,
                      RotacionFinal,traslation,pointsize=None,maxdepth=None):
    IRotFin=RotacionFinal.transpose()
    NTPOINTS=LIDARpoints.transpose()-traslation
    CFLidarPoints=np.matmul(IRotFin,NTPOINTS).T
    xypoints=np.zeros((CFLidarPoints.shape))
    for o,ENpoints in enumerate(CFLidarPoints):
        xypp,_=cv2.projectPoints(np.float32([[0.,0.,0.]]),ENpoints*0,ENpoints,camera_matrix,dist_coefs)
        xypoints[o]=np.hstack([xypp.squeeze(),1])
        

    imagesize=image.shape
    Validbool=(xypoints[:,0]<imagesize[1]) & (xypoints[:,0]>0) & (xypoints[:,1]<imagesize[0]) & (xypoints[:,1]>0)
    validpoints=np.int16(xypoints[Validbool])
    validdepths=CFLidarPoints[Validbool]

    LIDARIMAGE=np.zeros((imagesize),dtype =np.uint8)
    if maxdepth is None:
        ecuDEPTH=(validdepths[:,2]-np.min(validdepths[:,2]))/(np.max(validdepths[:,2])-np.min(validdepths[:,2]))
    else:
        ecuDEPTH=np.minimum(validdepths[:,2],maxdepth)/maxdepth
    for i,point in enumerate(validpoints):
        if pointsize is None:
            LIDARIMAGE[point[1],point[0],:]=np.uint8(np.multiply(cm.jet(ecuDEPTH[i]),255))[:3]
        else:
            col=np.multiply(cm.jet(ecuDEPTH[i]),255)[:3]
            center=point[0],point[1]
            cv2.circle(LIDARIMAGE,center,pointsize,color=col,thickness =-1)

    mask=cv2.bitwise_not(LIDARIMAGE[:,:,0])
    maskedim=cv2.bitwise_and(image,image,mask=mask)
    finalim=cv2.add(maskedim,LIDARIMAGE)
    return finalim,LIDARIMAGE

def PointProject(image,camera_matrix,Points,pointsize=None):
    xypoints=np.zeros((Points.shape))
    for o,ENpoints in enumerate(Points):
        nodepthpoints=ENpoints/ENpoints[2]
        xypoints[o]=np.int16(np.matmul(camera_matrix,nodepthpoints))

    imagesize=image.shape
    Validbool=(xypoints[:,0]<imagesize[1]) & (xypoints[:,0]>0) & (xypoints[:,1]<imagesize[0]) & (xypoints[:,1]>0)
    validpoints=np.int16(xypoints[Validbool])
    validdepths=Points[Validbool]

    LIDARIMAGE=np.zeros((imagesize),dtype =np.uint8)
    ecuDEPTH=(validdepths[:,2]-np.min(validdepths[:,2]))/(np.max(validdepths[:,2])-np.min(validdepths[:,2]))
    for i,point in enumerate(validpoints):
        if pointsize is None:
            LIDARIMAGE[point[1],point[0],:]=np.uint8(np.multiply(cm.jet(ecuDEPTH[i]),255))[:3]
        else:
            col=np.multiply(cm.jet(ecuDEPTH[i]),255)[:3]
            center=point[0],point[1]
            cv2.circle(LIDARIMAGE,center,pointsize,col,-1)

    mask=cv2.bitwise_not(LIDARIMAGE[:,:,0])
    maskedim=cv2.bitwise_and(image,image,mask=mask)
    finalim=cv2.add(maskedim,LIDARIMAGE)
    return finalim

def open3dpaint(nppoints,select=None,axis=None,axissize=0.5,color = 'jet'):
    assert((type(select) is list) | (type(select) is tuple) | (select is None) )
    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]
    if select is not None:
        assert(max(select)<len(nppoints))
        assert(min(select)>=0)
    assert len(nppoints[0])>0, 'Containes 0 Points'
    try:
        vis = open3d.Visualizer()
        vis.create_window()
        if select is not None:
            for i in select:
                points = convertcloud(nppoints[i])
                colNORM = i/len(select)/2 + i%2*.5
                if type(color) == np.ndarray:
                    pass
                elif color == 'jet':
                    color=cm.jet(colNORM)[:3]
                else:
                    color=cm.Set1(colNORM)[:3]
                points.colors = open3d.Vector3dVector(np.ones_like(i)*color)
                vis.add_geometry(points)
        else:
            for n,i in enumerate(nppoints):
                points=convertcloud(i)
                colNORM = n/len(nppoints)/2 + n%2*.5
                if type(color) == np.ndarray:
                    pass
                elif color == 'jet':
                    color=cm.jet(colNORM)[:3]
                else:
                    color=cm.Set1(colNORM)[:3]
                points.colors = open3d.Vector3dVector(np.ones_like(i)*color)
                vis.add_geometry(points)
        if axis is not None:
            if type(axis) is list:
                mesh_frame = open3d.create_mesh_coordinate_frame(size = axissize, origin = axis)
            else:
                mesh_frame = open3d.create_mesh_coordinate_frame(size = axissize, origin = [0, 0, 0])
            vis.add_geometry(mesh_frame)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        vis.destroy_window()
        print(e)
        
        
def open3dselect(nppoints, color=None):
    assert len(nppoints[0])>0, 'Containes 0 Points'
    try:
        vis = open3d.VisualizerWithEditing()
        vis.create_window()
        points=convertcloud(nppoints)
        if color is not None:
            points.colors = open3d.Vector3dVector(np.ones_like(nppoints)*color)
        vis.add_geometry(points)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        vis.destroy_window()
        print(e)
        
def getPrincipalVectors(A): #get pricipal vectors and values of a matrix centered around (0,0,0)
    VT=np.linalg.eig(np.matmul(A.T,A))
    sort = sorted(zip(VT[0],VT[1].T.tolist()),reverse=True)
    Values,Vectors = zip(*sort)
    return Vectors,Values

def ExtrinsicTransform(points,rotation,traslation): #Transform points with given rotation/traslation
    return (np.matmul(rotation,points.transpose())+traslation).transpose()

def Dpoint2Plane(point,planecoefs): #get minimum distance from a point to a plane
    return (planecoefs[0]*point[0]+planecoefs[1]*point[1]+planecoefs[2]*point[2]+planecoefs[3])/np.linalg.norm(planecoefs[0:3])

def DistPoint2Line(point,linepoint1, linepoint2=np.array([0,0,0])): #get minimum destance from a point to a line
    return np.linalg.norm(np.cross((point-linepoint2),(point-linepoint1)))/np.linalg.norm(linepoint1 - linepoint2)

def point2azimuth(point): #get the azimuth of a point from its x,y coordinates
    if point[1]==0:
        return np.pi/2 - np.sign(point[0]) * np.pi/2
    elif point[1]>0:
        return np.pi / 2 - np.arctan(point[0]/point[1])
    else:
        return np.pi * 3 / 2 - np.arctan(point[0]/point[1])
    
def point2azimuthVELO(point): #get the azimuth of a point from its x,y coordinates
    if point[0]==0:
        return np.pi/2 - np.sign(point[1]) * np.pi/2
    elif point[0]>0:
        return np.pi / 2 - np.arctan(point[1]/point[0])
    else:
        return np.pi * 3 / 2 - np.arctan(point[1]/point[0])

    
    
#Old #########################################################
def notnow():
    #Custom ransac
    inputpoints = planepoints[:,0:2]
    outputpoints = planepoints[:,2]
    s = int(len(inputpoints)*0.05)
    N = 20
    d = 0.05
    T = int(len(inputpoints)*0.01)
    bestcost=[]
    bestcoefs=[]
    usever = True
    for i in range(N):
        choices = np.random.choice(len(inputpoints),s,replace = False)
        X = inputpoints[choices,0]
        Y = inputpoints[choices,1]
        Z = outputpoints[choices]
        x0=[0,0,0]
        res = scipy.optimize.least_squares(planeverticalequfit,x0,args=(X,Y,Z),max_nfev=10)
        #res = scipy.optimize.least_squares(planeequfit,x0,args=(X,Y,Z),max_nfev=10)
        coefs = res.x
        cost = res.cost
        goodpoints = [i for i in np.vstack([X,Y,Z]).T if Dpoint2Plane(i,np.hstack([coefs[0:2],-1,coefs[2]]))<d]
        if len(goodpoints)>T:
            X = goodpoints[0]
            Y = goodpoints[1]
            Z = goodpoints[2]
            res = scipy.optimize.least_squares(planeverticalequfit,x0,args=(X,Y,Z),max_nfev=10)
            #res = scipy.optimize.least_squares(planeequfit,x0,args=(X,Y,Z),max_nfev=10)
            coefs = res.x
            cost = res.cost
        bestcoefs.append(coefs)
        bestcost.append(cost)
    index = bestcost.index(min(bestcost))
    coef = bestcoefs[index]
    cost = bestcost[index]
    inliermask = [Dpoint2Plane(i,np.hstack([coef[0:2],-1,coef[2]]))<d for i in planepoints]
    

def FindLaserPlanes(inputpoints ,samplesize = 0.02, goodthresh = 0.9, overlap = 3 ,disthesh = 0.03, ):
    if samplesize<1:
        s = int(len(inputpoints)*samplesize)
    else:
        s = samplesize
    newstart = int(s/overlap)
    n = 0
    savedlines = []
    savedcentroids = []
    savedpoints = []
    savedvectors = []
    savedlength = []
    foundline = False
    i = s
    first = 0
    while i < len(inputpoints):
        choisepoints = inputpoints[ first : i ]
        centroid = np.sum(choisepoints,0)/len(choisepoints)
        Centeredchoisepoints = choisepoints - centroid
        if foundline == False:
            U,S,VT=np.linalg.svd(Centeredchoisepoints)
            linevector = VT[0]
        goodpointsline = [i + centroid for i in Centeredchoisepoints if DistPoint2Line(i,linevector)<disthesh]
        if len(goodpointsline)/len(choisepoints)>goodthresh:  #good found
            if foundline == False:
                oldlinevector = linevector
            foundline = True
            if angle_b_vectors(oldlinevector,linevector)>0.1:
                Usaved = np.unique(savedpoints,axis=0)
                savedlines.append(Usaved)
                savecentroid = np.sum(Usaved,0)/len(Usaved)
                savedcentroids.append(savecentroid)
                savedvectors.append(oldlinevector)
                savedlength.append(np.linalg.norm(savedpoints[-1]-savedpoints[0]))
                savedpoints = []
                foundline=False
            else:
                savedpoints = savedpoints + goodpointsline
        elif foundline == True:
            Usaved = np.unique(savedpoints,axis=0)
            savedlines.append(Usaved)
            savecentroid = np.sum(Usaved,0)/len(Usaved)
            savedcentroids.append(savecentroid)
            savedvectors.append(oldlinevector)
            savedlength.append(np.linalg.norm(savedpoints[-1]-savedpoints[0]))
            savedpoints = []
            foundline=False
        else:
            foundline=False
        n += 1
        first = newstart * n
        i = s + first
    if foundline == True:
        Usaved = np.unique(savedpoints,axis=0)
        savedlines.append(Usaved)
        savecentroid = np.sum(Usaved,0)/len(Usaved)
        savedcentroids.append(savecentroid)
        savedvectors.append(oldlinevector)
        savedlength.append(np.linalg.norm(savedpoints[-1]-savedpoints[0]))
    return np.array(savedlines),np.array(savedcentroids),np.array(savedvectors),np.array(savedlength)


def linemodel(x,coefs):
    return x*coefs[0]+coefs[1]

def planemodel(x,y,coefs):
    return x*coefs[0]+y*coefs[1]+coefs[2]

def lineequfit(coefs,ins,outs):
    return outs-linemodel(ins,coefs)

def planeequfit(coefs,x,y,outs):
    return outs-planemodel(x,y,coefs)

def planeverticalequfit(coefs,x,y,outs):
    return abs(outs-planemodel(x,y,coefs))+abs(10/(coefs[0]+0.00001))+abs(10/(coefs[1]+0.00001))


def SVDLstSqr(X,Y):
    U,s,VT=np.linalg.svd(X)
    S = np.diag(s)
    Si=np.zeros([U.shape[1],VT.shape[0]])
    Si[0:S.shape[0],0:S.shape[0]]=np.diag(1/s)
    return np.matmul(VT.T,np.matmul(Si.T,np.matmul(U.T,Y)))



def SVDRigidBodyTransform(Points1,points2):
    centroid1 = np.sum(Points1,0)/len(Points1)
    centroid2 = np.sum(points2,0)/len(points2)
    CenteredVector1=(Points1-centroid1).transpose()
    CenteredVector2=(points2-centroid2).transpose()

    YiT=CenteredVector2.transpose()
    Xi=CenteredVector1
    Wi=np.eye(Xi.shape[1])

    S=np.matmul(Xi,np.matmul(Wi,YiT))
    U,SigNDiag,VT=np.linalg.svd(S, full_matrices=True)
    Raro=np.eye(3)
    Raro[-1,-1]=np.linalg.det(np.matmul(VT.transpose(),U.transpose()))

    RotacionFinal=np.matmul(VT.transpose(),np.matmul(Raro,U.transpose()))
    traslation=(centroid2-np.matmul(RotacionFinal,centroid1)).reshape(3,1)
    return traslation,RotacionFinal


def rotation_matrix_from_vectors(vec1, vec2):
    if all(np.abs(vec1)==np.abs(vec2)):
        return np.eye(3)
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def makecylinder(model=[0,0,0,1,0,0,1],length = 1,dense=10):
    radius = model[6]
    X,Y,Z = model[:3]
    direction = model[3:6]/np.linalg.norm(model[3:6])
    n = np.arange(0,360,int(360/dense))
    height = np.arange(0,length,length/dense)
    n = np.deg2rad(n)
    x,z = np.meshgrid(n,height)
    x = x.flatten()
    z = z.flatten()
    cyl = np.vstack([np.cos(x)*radius,np.sin(x)*radius,z]).T
    rotation = rotation_matrix_from_vectors([0,0,1],model[3:6])
    rotatedcyl = np.matmul(rotation,cyl.T).T + np.array([X,Y,Z])
    return rotatedcyl   

def makeplane(coefs,center=[0,0,0]):
    ll, ul = -1,1
    step = (ul-ll)/10
    planepoints = np.meshgrid(np.arange(ll,ul,step),np.arange(ll,ul,step))
    plane = np.array([planepoints[0].flatten(),planepoints[1].flatten(),np.zeros(len(planepoints[0].flatten()))]).T
    R = rotation_matrix_from_vectors([0,0,1],coefs[0:3])
    return np.add(np.matmul(R,plane.T).T,center)



def PCL3dpaint(nppoints,axis=None):
    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]
    try:
        newv = pclpy.pcl.visualization.PCLVisualizer()
        if axis is not None:
            newv.addCoordinateSystem(axis);
        for n,i in enumerate(nppoints):
            if type(i) == pclpy.pcl.PointCloud.PointXYZRGB:
                newv.addPointCloud(i,'PC'+str(n))
            elif type(i) == pclpy.pcl.PointCloud.PointXYZ:
                colNORM = n / len(nppoints) / 2 + n % 2 * .5
                color = np.uint8( np.array( cm.jet( colNORM)[:3]) * 255)
                colorHandeler = pclpy.pcl.visualization.PointCloudColorHandlerCustom.PointXYZ(i,color[0],color[1],color[2])
                newv.addPointCloud(i,colorHandeler,'PC'+str(n))
            else:
                colNORM = n / len(nppoints) / 2 + n % 2 * .5
                color = np.uint8( np.array( cm.jet( colNORM)[:3]) * 255)
                pointcloud = pclpy.pcl.PointCloud.PointXYZ(i)
                colorHandeler = pclpy.pcl.visualization.PointCloudColorHandlerCustom.PointXYZ(pointcloud,color[0],color[1],color[2])
                newv.addPointCloud(pointcloud,colorHandeler,'PC'+str(n))

        while not newv.wasStopped():
            newv.spinOnce(10)
        newv.close()
        
    except Exception as e:
        newv.close()
        print(e)