import cv2
from cv2 import aruco as aruco
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import glob
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import math
import os
from sklearn import linear_model
import Corners
import Plane
from pdb import set_trace as Debug

def getcharcornersMULTI(images,cameradict,arucodict,charboard,equalize=0,minim = 48):
    allCorners  = [] # 3d point in real world space
    allIds  = []
    allDirs  = []

    paramiters=cv2.aruco.DetectorParameters_create()
    paramiters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    paramiters.adaptiveThreshWinSizeMax = 301
    paramiters.adaptiveThreshWinSizeStep = 102

    for decimator,fname in enumerate(images[:]):
        imgc = cv2.imread(fname,0)
        imgc = cv2.resize(imgc,(cameradict['seedW'],cameradict['seedH']))
        
        if equalize != 0:
            imgc = cv2.min(imgc,equalize)
            imgc = cv2.equalizeHist(imgc)
            
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(imgc, arucodict,parameters =paramiters)
        if len(corners)>2:
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,imgc,charboard)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>=minim:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])
                    allDirs.append(decimator)
                    
    return allCorners, allIds, allDirs


def findarucoSimpleMULTI(images,cameradict,arucodict,charboard,equalize=0,minim = 30):
    allCorners  = [] # 3d point in real world space
    allIds  = []
    allDirs  = []
    rots = []
    tras = []
    impoints = []

    paramiters=cv2.aruco.DetectorParameters_create()
    paramiters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    paramiters.adaptiveThreshWinSizeMax = 301
    paramiters.adaptiveThreshWinSizeStep = 102

    for decimator,fname in enumerate(images[:]):
        imgc = cv2.imread(fname,0)
        imgc = cv2.resize(imgc,(cameradict['seedW'],cameradict['seedH']))
        
        if equalize != 0:
            imgc = cv2.min(imgc,equalize)
            imgc = cv2.equalizeHist(imgc)
            
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(imgc, arucodict,parameters =paramiters)
        if len(corners)>2:
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,imgc,charboard)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>=minim:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])
                    allDirs.append(decimator)
                    rots1 = np.ones([1,3])
                    tras2 = np.ones([1,3])
                    trans = cv2.aruco.estimatePoseCharucoBoard(res2[1], res2[2], charboard, cameradict['camera_matrix'], cameradict['dist_coefs'],rots1,tras2)
                    rots1 = trans[1]
                    tras2 = trans[2]
                    rots.append(rots1)
                    tras.append(tras2)
                    impointsPar,_=cv2.projectPoints(charboard.chessboardCorners,rots1,tras2,cameradict['camera_matrix'],cameradict['dist_coefs'])
                    impoints.append(impointsPar)
    return rots, tras, allCorners, allIds, allDirs, impoints

def find3arucosSimple(frame,camera_matrix,dist_coefs, equalize = 0):
    arucodict1=aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    arucodict2=aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    arucodict3=aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    markersize = 0.0848
    squaresize = 0.1062
    charboard1=aruco.CharucoBoard_create(9,7,squaresize,markersize,arucodict1) 
    charboard2=aruco.CharucoBoard_create(9,7,squaresize,markersize,arucodict2) 
    charboard3=aruco.CharucoBoard_create(9,7,squaresize,markersize,arucodict3) 
    paramiters=cv2.aruco.DetectorParameters_create()
    paramiters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    paramiters.adaptiveThreshWinSizeMax = 301
    paramiters.adaptiveThreshWinSizeStep = 102

    wframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if equalize != 0:
        wframe = cv2.min(wframe,equalize)
        wframe = cv2.equalizeHist(wframe)
        
    Brvec1 = np.zeros([3,1],dtype = np.float32)
    Btvec1 = np.zeros([3,1],dtype = np.float32)
    Brvec2 = np.zeros([3,1],dtype = np.float32)
    Btvec2 = np.zeros([3,1],dtype = np.float32)
    Brvec3 = np.zeros([3,1],dtype = np.float32)
    Btvec3 = np.zeros([3,1],dtype = np.float32)
    findingaruco1,ids1,reg1=aruco.detectMarkers(wframe,arucodict1,parameters =paramiters)
    if len(findingaruco1)>0:
        findingaruco2,ids2,reg2=aruco.detectMarkers(wframe,arucodict2,parameters =paramiters)
        if len(findingaruco2)>0:
            findingaruco3,ids3,reg3=aruco.detectMarkers(wframe,arucodict3,parameters =paramiters)
    IntCor1 = cv2.aruco.interpolateCornersCharuco(findingaruco1,ids1,wframe,charboard1,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
    cv2.aruco.estimatePoseCharucoBoard(IntCor1[1], IntCor1[2], charboard1, camera_matrix, dist_coefs,Brvec1,Btvec1)

    IntCor2 = cv2.aruco.interpolateCornersCharuco(findingaruco2,ids2,wframe,charboard2,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
    cv2.aruco.estimatePoseCharucoBoard(IntCor2[1], IntCor2[2], charboard2, camera_matrix, dist_coefs,Brvec2,Btvec2)

    IntCor3 = cv2.aruco.interpolateCornersCharuco(findingaruco3,ids3,wframe,charboard3,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
    cv2.aruco.estimatePoseCharucoBoard(IntCor3[1], IntCor3[2], charboard3, camera_matrix, dist_coefs,Brvec3,Btvec3)
    return Brvec1,Btvec1,Brvec2,Btvec2,Brvec3,Btvec3

def get3arucosextrinsics(frame,camera_matrix,dist_coefs , equalize=None):
    Brvec1,Btvec1,Brvec2,Btvec2,Brvec3,Btvec3 = find3arucosSimple(frame,camera_matrix,dist_coefs , equalize)
    rotationmat,_=cv2.Rodrigues(Brvec1)
    extrinsicmatCam1 = np.hstack([rotationmat,Btvec1])
    rotationmat,_=cv2.Rodrigues(Brvec2)
    extrinsicmatCam2 = np.hstack([rotationmat,Btvec2])
    rotationmat,_=cv2.Rodrigues(Brvec3)
    extrinsicmatCam3 = np.hstack([rotationmat,Btvec3])
    return extrinsicmatCam1,extrinsicmatCam2,extrinsicmatCam3

def find3arucosSimpleDraw(frameorig,camera_matrix,dist_coefs, equalize = 0):
    frame = frameorig.copy()
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
    paramiters.adaptiveThreshWinSizeMax = 301
    paramiters.adaptiveThreshWinSizeStep = 102

    
    wframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if equalize != 0:
        wframe = cv2.min(wframe,equalize)
        wframe = cv2.equalizeHist(wframe)

    Brvec1 = np.zeros([3,1],dtype = np.float32)
    Btvec1 = np.zeros([3,1],dtype = np.float32)
    Brvec2 = np.zeros([3,1],dtype = np.float32)
    Btvec2 = np.zeros([3,1],dtype = np.float32)
    Brvec3 = np.zeros([3,1],dtype = np.float32)
    Btvec3 = np.zeros([3,1],dtype = np.float32)
    
    findingaruco1,ids1,reg1=aruco.detectMarkers(wframe,arucodict1,parameters =paramiters)
    if len(findingaruco1)>0:
        findingaruco2,ids2,reg2=aruco.detectMarkers(wframe,arucodict2,parameters =paramiters)
        if len(findingaruco2)>0:
            findingaruco3,ids3,reg3=aruco.detectMarkers(wframe,arucodict3,parameters =paramiters)
    IntCor1 = cv2.aruco.interpolateCornersCharuco(findingaruco1,ids1,wframe,charboard1,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
    cv2.aruco.estimatePoseCharucoBoard(IntCor1[1], IntCor1[2], charboard1, camera_matrix, dist_coefs,Brvec1,Btvec1)
    IntCor2 = cv2.aruco.interpolateCornersCharuco(findingaruco2,ids2,wframe,charboard2,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
    cv2.aruco.estimatePoseCharucoBoard(IntCor2[1], IntCor2[2], charboard2, camera_matrix, dist_coefs,Brvec2,Btvec2)
    IntCor3 = cv2.aruco.interpolateCornersCharuco(findingaruco3,ids3,wframe,charboard3,cameraMatrix =camera_matrix,distCoeffs =dist_coefs)
    cv2.aruco.estimatePoseCharucoBoard(IntCor3[1], IntCor3[2], charboard3, camera_matrix, dist_coefs,Brvec3,Btvec3)
    
    dectectim = aruco.drawDetectedMarkers(frame,findingaruco1,ids1)
    dectectim = aruco.drawDetectedMarkers(dectectim,findingaruco2,ids2)
    dectectim = aruco.drawDetectedMarkers(dectectim,findingaruco3,ids3)
    
    final = aruco.drawAxis(dectectim,camera_matrix,dist_coefs,Brvec1,Btvec1,markersize)
    final = aruco.drawAxis(final,camera_matrix,dist_coefs,Brvec2,Btvec2,markersize)
    final = aruco.drawAxis(final,camera_matrix,dist_coefs,Brvec3,Btvec3,markersize)

    return final

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


def findbestmatch(info,alldist,allangles):
    best = [0,1,2]
    bestglobal = 100
    a_factor = 4
    for n,i in enumerate(info):
        for n2,i2 in enumerate(info):
            if n==n2:
                continue
            bufanlge12 = Corners.angle_b_vectors(i[1][2],i2[1][2])
            if bufanlge12 > np.pi/2:
                bufanlge12 = np.pi - bufanlge12
            bufdist12 = np.linalg.norm(i[2]-i2[2])
            for n3,i3 in enumerate(info):
                if (n3 == n) | (n3 == n2):
                    continue
                bufanlge13 = Corners.angle_b_vectors(i[1][2],i3[1][2]) 
                if bufanlge13 > np.pi/2:
                    bufanlge13 = np.pi - bufanlge13
                bufanlge23 = Corners.angle_b_vectors(i2[1][2],i3[1][2]) 
                if bufanlge23 > np.pi/2:
                    bufanlge23 = np.pi - bufanlge23
                bufdist13 = np.linalg.norm(i[2]-i3[2])
                bufdist23 = np.linalg.norm(i2[2]-i3[2])

                distresult1 = np.linalg.norm(alldist-np.array([bufdist12,bufdist13,bufdist23]))
                angleresult1 = np.linalg.norm(allangles-np.array([bufanlge12,bufanlge13,bufanlge23]))
                final1 = distresult1**2 + angleresult1**2

                distresult2 = np.linalg.norm(alldist-np.array([bufdist13,bufdist23,bufdist12]))
                angleresult2 = np.linalg.norm(allangles-np.array([bufanlge13,bufanlge23,bufanlge12]))
                final2 = distresult2**2 + angleresult2**2

                distresult3 = np.linalg.norm(alldist-np.array([bufdist23,bufdist12,bufdist13]))
                angleresult3 = np.linalg.norm(allangles-np.array([bufanlge23,bufanlge12,bufanlge13]))
                final3 = distresult3**2 + angleresult3**2

                distresult4 = np.linalg.norm(alldist-np.array([bufdist12,bufdist23,bufdist13]))
                angleresult4 = np.linalg.norm(allangles-np.array([bufanlge12,bufanlge23,bufanlge13]))
                final4 = distresult4**2 + angleresult4**2

                distresult5 = np.linalg.norm(alldist-np.array([bufdist13,bufdist12,bufdist23]))
                angleresult5 = np.linalg.norm(allangles-np.array([bufanlge13,bufanlge12,bufanlge23]))
                final5 = distresult5**2 + angleresult5**2

                distresult6 = np.linalg.norm(alldist-np.array([bufdist23,bufdist13,bufdist12]))
                angleresult6 = np.linalg.norm(allangles-np.array([bufanlge23,bufanlge13,bufanlge12]))
                final6 = distresult6**2 + angleresult6**2

                bestlocal = min([final1,final2,final3,final4,final5,final6])
                if bestlocal < bestglobal:
                    bestglobal = bestlocal
                    org = [final1,final2,final3,final4,final5,final6].index(bestlocal)
                    if org == 0:
                        best = [n,n2,n3]
                    if org == 1:
                        best = [n3,n,n2]
                    if org == 2:
                        best = [n2,n3,n]
                    if org == 3:
                        best = [n2,n,n3]
                    if org == 4:
                        best = [n,n3,n2]
                    if org == 5:
                        best = [n3,n2,n]
    return best,bestglobal



def get3arucodescriptor(extrinsicmatCam1,extrinsicmatCam2,extrinsicmatCam3):
    squaresize = 0.1062
    
    x=np.arange(10)
    y=np.arange(8)
    z=np.zeros([8,10])
    x,y=np.meshgrid(x,y)
    cornerpoints=np.array([x.flatten(),y.flatten(),z.flatten()],dtype=np.float32)*squaresize
    cornerpointsH = np.vstack([cornerpoints,np.ones_like(cornerpoints[0,:])])
    puntos1Camara = np.matmul(extrinsicmatCam1,cornerpointsH).T
    puntos2Camara = np.matmul(extrinsicmatCam2,cornerpointsH).T
    puntos3Camara = np.matmul(extrinsicmatCam3,cornerpointsH).T
    #CCpuntosCamara = np.vstack((puntos1Camara,puntos2Camara,puntos3Camara))
    centroid1 = np.mean(puntos1Camara,axis=0)
    centroid2 = np.mean(puntos2Camara,axis=0)
    centroid3 = np.mean(puntos3Camara,axis=0)
    dist12 = np.linalg.norm(centroid1-centroid2)
    dist13 = np.linalg.norm(centroid1-centroid3)
    dist23 = np.linalg.norm(centroid2-centroid3)
    alldist = np.hstack([dist12,dist13,dist23])

    norm1=np.matmul(extrinsicmatCam1[0:3,0:3],np.array([0,0,1]))
    norm2=np.matmul(extrinsicmatCam2[0:3,0:3],np.array([0,0,1]))
    norm3=np.matmul(extrinsicmatCam3[0:3,0:3],np.array([0,0,1]))

    anlge12 = Corners.angle_b_vectors(norm1,norm2)
    anlge13 = Corners.angle_b_vectors(norm1,norm3)
    anlge23 = Corners.angle_b_vectors(norm2,norm3)
    allangles = np.hstack([anlge12,anlge13,anlge23])
    return alldist,allangles



def get3arucosearchpoints(extrinsicmatCam1, extrinsicmatCam2, extrinsicmatCam3, lidarPlane1, lidarPlane2, lidarPlane3):
    squaresize = 0.1062
    x=np.arange(10)
    y=np.arange(8)
    z=np.zeros([8,10])
    x,y=np.meshgrid(x,y)
    cornerpoints=np.array([x.flatten(),y.flatten(),z.flatten()],dtype=np.float32)*squaresize
    cornerpointsH = np.vstack([cornerpoints,np.ones_like(cornerpoints[0,:])])
    puntos1Camara = np.matmul(extrinsicmatCam1,cornerpointsH).T
    puntos2Camara = np.matmul(extrinsicmatCam2,cornerpointsH).T
    puntos3Camara = np.matmul(extrinsicmatCam3,cornerpointsH).T
    
    
    norm1 = np.matmul(extrinsicmatCam1[0:3,0:3],np.array([0,0,1]))
    norm2 = np.matmul(extrinsicmatCam2[0:3,0:3],np.array([0,0,1]))
    norm3 = np.matmul(extrinsicmatCam3[0:3,0:3],np.array([0,0,1]))
    Camera1Coefs = np.hstack([norm1,np.sum(norm1*extrinsicmatCam1[:,3])])
    Camera2Coefs = np.hstack([norm2,np.sum(norm2*extrinsicmatCam2[:,3])])
    Camera3Coefs = np.hstack([norm3,np.sum(norm3*extrinsicmatCam3[:,3])])

    lidarPlane1Good,_,_ = Plane.findplane(lidarPlane1)
    _,LidarPC1,LidarCenter1 = Plane.getPrincipalComponents(lidarPlane1Good)
    Lidar1Coefs = np.hstack([LidarPC1[2],np.sum(LidarPC1[2]*LidarCenter1)])
    print(len(lidarPlane1Good),len(lidarPlane1))

    lidarPlane2Good,_,_ = Plane.findplane(lidarPlane2)
    _,LidarPC2,LidarCenter2 = Plane.getPrincipalComponents(lidarPlane2Good)
    Lidar2Coefs = np.hstack([LidarPC2[2],np.sum(LidarPC2[2]*LidarCenter2)])
    print(len(lidarPlane2Good),len(lidarPlane2))

    lidarPlane3Good,_,_ = Plane.findplane(lidarPlane3)
    _,LidarPC3,LidarCenter3 = Plane.getPrincipalComponents(lidarPlane3Good)
    Lidar3Coefs = np.hstack([LidarPC3[2],np.sum(LidarPC3[2]*LidarCenter3)])
    print(len(lidarPlane3Good),len(lidarPlane3))

    planecoefsLIDAR = np.zeros((3,3))
    planecoefsLIDAR[0,:] = Lidar1Coefs[0:3]
    planecoefsLIDAR[1,:] = Lidar2Coefs[0:3]
    planecoefsLIDAR[2,:] = Lidar3Coefs[0:3]
    VertexResultLIDAR=np.array([Lidar1Coefs[3],Lidar2Coefs[3],Lidar3Coefs[3]])
    VertexLIDAR = np.linalg.solve(planecoefsLIDAR, VertexResultLIDAR)

    LidarNormal1Vect = Plane.makepointvector(LidarPC1[2],LidarCenter1)
    LidarNormal2Vect = Plane.makepointvector(LidarPC2[2],LidarCenter2)
    LidarNormal3Vect = Plane.makepointvector(LidarPC3[2],LidarCenter3)

    planecoefsCamera = np.zeros((3,3))
    planecoefsCamera[0,:] = Camera1Coefs[0:3]
    planecoefsCamera[1,:] = Camera2Coefs[0:3]
    planecoefsCamera[2,:] = Camera3Coefs[0:3]
    VertexResultCamera = np.array([Camera1Coefs[3],Camera2Coefs[3],Camera3Coefs[3]])
    VertexCamera = np.linalg.solve(planecoefsCamera, VertexResultCamera)

    CameraNormal1Vect = Plane.makepointvector(norm1,extrinsicmatCam1[:,3])
    CameraNormal2Vect = Plane.makepointvector(norm2,extrinsicmatCam2[:,3])
    CameraNormal3Vect = Plane.makepointvector(norm3,extrinsicmatCam3[:,3])

    Cintersect12=np.cross(norm1,norm2)
    Cintersect12 = Cintersect12/np.linalg.norm(Cintersect12)
    Cintersect23=np.cross(norm2,norm3)
    Cintersect23 = Cintersect23/np.linalg.norm(Cintersect23)
    Cintersect13=np.cross(norm1,norm3)
    Cintersect13 = Cintersect13/np.linalg.norm(Cintersect13)
    intersect12=np.cross(LidarPC1[2],LidarPC2[2])
    intersect12 = intersect12/np.linalg.norm(intersect12)
    intersect23=np.cross(LidarPC2[2],LidarPC3[2])
    intersect23 = intersect23/np.linalg.norm(intersect23)
    intersect13=np.cross(LidarPC1[2],LidarPC3[2])
    intersect13 = intersect13/np.linalg.norm(intersect13)

    PCameraInter12Vect = Plane.makepointvector(Cintersect12,VertexCamera)
    NCameraInter12Vect = Plane.makepointvector(-1*Cintersect12,VertexCamera)
    distanceP = np.sum(np.linalg.norm(np.vstack((puntos1Camara,puntos2Camara))-PCameraInter12Vect[-1],axis = 1))
    distanceN = np.sum(np.linalg.norm(np.vstack((puntos1Camara,puntos2Camara))-NCameraInter12Vect[-1],axis = 1))    

    if distanceP<distanceN:
        CameraInter12Vect = PCameraInter12Vect
    else:
        CameraInter12Vect = NCameraInter12Vect


    PCameraInter23Vect = Plane.makepointvector(Cintersect23,VertexCamera)
    NCameraInter23Vect = Plane.makepointvector(-1*Cintersect23,VertexCamera)
    distanceP = np.sum(np.linalg.norm(np.vstack((puntos2Camara,puntos3Camara))-PCameraInter23Vect[-1],axis = 1))
    distanceN = np.sum(np.linalg.norm(np.vstack((puntos2Camara,puntos3Camara))-NCameraInter23Vect[-1],axis = 1))
    if distanceP<distanceN:
        CameraInter23Vect = PCameraInter23Vect
    else:
        CameraInter23Vect = NCameraInter23Vect


    PCameraInter13Vect = Plane.makepointvector(Cintersect13,VertexCamera)
    NCameraInter13Vect = Plane.makepointvector(-1*Cintersect13,VertexCamera)
    distanceP = np.sum(np.linalg.norm(np.vstack((puntos1Camara,puntos3Camara))-PCameraInter13Vect[-1],axis = 1))
    distanceN = np.sum(np.linalg.norm(np.vstack((puntos1Camara,puntos3Camara))-NCameraInter13Vect[-1],axis = 1))
    if distanceP<distanceN:
        CameraInter13Vect = PCameraInter13Vect
    else:
        CameraInter13Vect = NCameraInter13Vect

    PLIDARInter12Vect = Plane.makepointvector(intersect12,VertexLIDAR)
    NLIDARInter12Vect = Plane.makepointvector(-1*intersect12,VertexLIDAR)
    distanceP = np.sum(np.linalg.norm(np.vstack((lidarPlane1,lidarPlane2))-PLIDARInter12Vect[-1],axis = 1))
    distanceN = np.sum(np.linalg.norm(np.vstack((lidarPlane1,lidarPlane2))-NLIDARInter12Vect[-1],axis = 1))
    if distanceP<distanceN:
        LIDARInter12Vect = PLIDARInter12Vect
    else:
        LIDARInter12Vect = NLIDARInter12Vect


    PLIDARInter23Vect = Plane.makepointvector(intersect23,VertexLIDAR)
    NLIDARInter23Vect = Plane.makepointvector(-1*intersect23,VertexLIDAR)
    distanceP = np.sum(np.linalg.norm(np.vstack((lidarPlane2,lidarPlane3))-PLIDARInter23Vect[-1],axis = 1))
    distanceN = np.sum(np.linalg.norm(np.vstack((lidarPlane2,lidarPlane3))-NLIDARInter23Vect[-1],axis = 1))
    if distanceP<distanceN:
        LIDARInter23Vect = PLIDARInter23Vect
    else:
        LIDARInter23Vect = NLIDARInter23Vect


    PLIDARInter13Vect = Plane.makepointvector(intersect13,VertexLIDAR)
    NLIDARInter13Vect = Plane.makepointvector(-1*intersect13,VertexLIDAR)
    distanceP = np.sum(np.linalg.norm(np.vstack((lidarPlane1,lidarPlane3))-PLIDARInter13Vect[-1],axis = 1))
    distanceN = np.sum(np.linalg.norm(np.vstack((lidarPlane1,lidarPlane3))-NLIDARInter13Vect[-1],axis = 1))
    if distanceP<distanceN:
        LIDARInter13Vect = PLIDARInter13Vect
    else:
        LIDARInter13Vect = NLIDARInter13Vect


    CameraSearchpoints=np.vstack([CameraInter12Vect[-1],CameraInter23Vect[-1],CameraInter13Vect[-1],
                           VertexCamera])
    LidarSearchpoints=np.vstack([LIDARInter12Vect[-1],LIDARInter23Vect[-1],LIDARInter13Vect[-1],
                           VertexLIDAR])
    return CameraSearchpoints,LidarSearchpoints



def get3arucoPlaneCoefs(extrinsicmatCam1, extrinsicmatCam2, extrinsicmatCam3, lidarPlane1, lidarPlane2, lidarPlane3):
    squaresize = 0.1062
    x=np.arange(10)
    y=np.arange(8)
    z=np.zeros([8,10])
    x,y=np.meshgrid(x,y)
    cornerpoints=np.array([x.flatten(),y.flatten(),z.flatten()],dtype=np.float32)*squaresize
    cornerpointsH = np.vstack([cornerpoints,np.ones_like(cornerpoints[0,:])])
    puntos1Camara = np.matmul(extrinsicmatCam1,cornerpointsH).T
    puntos2Camara = np.matmul(extrinsicmatCam2,cornerpointsH).T
    puntos3Camara = np.matmul(extrinsicmatCam3,cornerpointsH).T
    
    norm1 = np.matmul(extrinsicmatCam1[0:3,0:3],np.array([0,0,1]))
    norm2 = np.matmul(extrinsicmatCam2[0:3,0:3],np.array([0,0,1]))
    norm3 = np.matmul(extrinsicmatCam3[0:3,0:3],np.array([0,0,1]))
    Camera1Coefs = np.hstack([norm1,np.sum(norm1*extrinsicmatCam1[:,3])])
    Camera2Coefs = np.hstack([norm2,np.sum(norm2*extrinsicmatCam2[:,3])])
    Camera3Coefs = np.hstack([norm3,np.sum(norm3*extrinsicmatCam3[:,3])])

    lidarPlane1Good,_,_ = Plane.findplane(lidarPlane1)
    _,LidarPC1,LidarCenter1 = Plane.getPrincipalComponents(lidarPlane1Good)
    Lidar1Coefs = np.hstack([LidarPC1[2],np.sum(LidarPC1[2]*LidarCenter1)])

    lidarPlane2Good,_,_ = Plane.findplane(lidarPlane2)
    _,LidarPC2,LidarCenter2 = Plane.getPrincipalComponents(lidarPlane2Good)
    Lidar2Coefs = np.hstack([LidarPC2[2],np.sum(LidarPC2[2]*LidarCenter2)])

    lidarPlane3Good,_,_ = Plane.findplane(lidarPlane3)
    _,LidarPC3,LidarCenter3 = Plane.getPrincipalComponents(lidarPlane3Good)
    Lidar3Coefs = np.hstack([LidarPC3[2],np.sum(LidarPC3[2]*LidarCenter3)])

    
    return Camera1Coefs,Camera2Coefs,Camera3Coefs,Lidar1Coefs,Lidar2Coefs,Lidar3Coefs