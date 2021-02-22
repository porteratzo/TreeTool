import pclpy
import numpy as np
import pandas as pd
import Libraries.segTree as segTree
import Libraries.Utils as Utils
from ellipse import LsqEllipse

class Tree_tool():
    def __init__(self, pointcloud = None, Ksearch = 0.08):
        if not pointcloud is None:
            assert (type(pointcloud) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(pointcloud) == pclpy.pcl.PointCloud.PointXYZ) or (type(pointcloud) == np.ndarray), 'Not valid pointcloud'
            if (type(pointcloud) == np.ndarray):
                self.Pointcloud = pclpy.pcl.PointCloud.PointXYZ(pointcloud)
            else:
                self.Pointcloud = pointcloud
            self.Ksearch = Ksearch
    
    def set_pointcloud(self, pointcloud):
        if not pointcloud is None:
            assert (type(pointcloud) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(pointcloud) == pclpy.pcl.PointCloud.PointXYZ) or (type(pointcloud) == np.ndarray), 'Not valid pointcloud'
            if (type(pointcloud) == np.ndarray):
                self.Pointcloud = pclpy.pcl.PointCloud.PointXYZ(pointcloud)
            else:
                self.Pointcloud = pointcloud
    
    def Step_1_Remove_Floor(self):
        Nogroundpoints, Ground = segTree.FloorRemove(self.Pointcloud)
        self.NongroundCloud = pclpy.pcl.PointCloud.PointXYZ(Nogroundpoints)
        self.GroundCloud = pclpy.pcl.PointCloud.PointXYZ(Ground)
        
        
    def set_Ksearch(self, Ksearch):
        self.Ksearch = Ksearch  
        

    def Step_2_Normal_Filtering(self, verticalityThresh = 0.08, NonNANcurvatureThresh = 0.12):
        self.normals = segTree.ExtractNormals(self.NongroundCloud.xyz, self.Ksearch)

        nanmask = np.bitwise_not(np.isnan(self.normals.normals[:,0]))
        NonNANpoints = self.NongroundCloud.xyz[nanmask]
        NonNANnormals = self.normals.normals[nanmask]
        NonNANcurvature = self.normals.curvature[nanmask]
        verticality = np.dot(NonNANnormals,[[0],[0],[1]])
        mask = (verticality < verticalityThresh) & (-verticalityThresh < verticality)  #0.1
        maskC = (NonNANcurvature < NonNANcurvatureThresh)## 0.12
        Fmask = mask.ravel() & maskC.ravel()

        onlyhorizontalpoints = NonNANpoints[Fmask]
        onlyhorizontalnormals = NonNANnormals[Fmask]
        
        self.nonFilterednormals = NonNANnormals
        self.nonFilteredpoints = pclpy.pcl.PointCloud.PointXYZ(self.NongroundCloud.xyz[nanmask])
        
        self.filteredpoints = pclpy.pcl.PointCloud.PointXYZ(onlyhorizontalpoints)
        self.filterednormals = onlyhorizontalnormals
    
    def Step_3_Eucladean_Clustering(self, tol=0.1, minc=40, maxc=6000000):
        self.cluster_list = segTree.EucladeanClusterExtract(self.filteredpoints.xyz, tol = tol, minc = minc, maxc = maxc)
        
    def Step_4_Group_Stems(self, max_angle = 0.4):
        GroupStems = []
        bufferStems = self.cluster_list.copy()
        for n,p in enumerate(self.cluster_list):
            Centroid = np.mean(p, axis = 0)
            vT,S = Utils.getPrincipalVectors(p-Centroid)
            strieghtness = S[0]/(S[0]+S[1]+S[2])

            clustersDICT = {}
            clustersDICT['cloud'] = p
            clustersDICT['strieghtness'] = strieghtness
            clustersDICT['center'] = Centroid
            clustersDICT['direction'] = vT
            GroupStems.append(clustersDICT)

        bufferStems = [i['cloud'] for i in GroupStems]
        for treenumber1 in reversed(range(0,len(bufferStems))):
            for treenumber2 in reversed(range(0,treenumber1-1)):
                center1 = GroupStems[treenumber1]['center']
                center2 = GroupStems[treenumber2]['center']
                angle1 = GroupStems[treenumber1]['direction'][0]
                angle2 = GroupStems[treenumber2]['direction'][0]
                dist1 = Utils.DistPoint2Line(center2,angle1+center1,center1)
                dist2 = Utils.DistPoint2Line(center1,angle2+center2,center2)
                if (dist1 < max_angle) | (dist2 < max_angle):
                    bufferStems[treenumber2] = np.vstack([bufferStems[treenumber2],bufferStems.pop(treenumber1)])
                    break

        self.complete_Stems = bufferStems
        
    def Step_5_Get_Ground_Level_Trees( self, lowstems_Height = 5, cutstems_Height = 5):
        pointpart = self.GroundCloud.xyz
        
        A = np.c_[np.ones(pointpart.shape[0]), pointpart[:,:2], np.prod(pointpart[:,:2], axis=1), pointpart[:,:2]**2]
        self.Ground_Model_C,_,_,_ = np.linalg.lstsq(A, pointpart[:,2], rcond=None)

        StemsWithGround = []
        for i in self.complete_Stems:
            center = np.mean(i,0)
            X,Y = center[:2]
            Z = np.dot(np.c_[np.ones(X.shape), X, Y, X*Y, X**2, Y**2], self.Ground_Model_C)
            StemsWithGround.append([i,[X,Y,Z[0]]])

        lowStems = [i for i in StemsWithGround if np.min(i[0],axis=0)[2] < (lowstems_Height + i[1][2])]
        cutstems = [[i[0][i[0][:,2]<(cutstems_Height + i[1][2])],i[1]] for i in lowStems]
        
        
        self.cutstems = cutstems
        self.lowstems = [i[0] for i in cutstems]
        
    def Step_6_Get_Cylinder_Tree_Models( self, searchRadius = 0.1):
        finalstems = []
        stemcyls = []
        rech = []
        for p in self.cutstems:
            segpoints = p[0]
            indices, model = segTree.segment_normals(segpoints, searchRadius = searchRadius, model=pclpy.pcl.sample_consensus.SACMODEL_CYLINDER, method=pclpy.pcl.sample_consensus.SAC_RANSAC, normalweight=0.01, miter=10000, distance=0.08, rlim=[0,0.4])
            if len(indices)>10:
                if abs(np.dot(model[3:6],[0,0,1])/np.linalg.norm(model[3:6])) > 0.5:
                    newmodel = np.array(model)
                    Z = 1.3 + p[1][2]
                    Y = model[1] + model[4] * (Z - model[2]) / model[5]
                    X = model[0] + model[3] * (Z - model[2]) / model[5]
                    newmodel[0:3] = np.array([X,Y,Z])
                    newmodel[3:6] = Utils.similarize(newmodel[3:6],[0,0,1])
                    finalstems.append({'tree':segpoints[indices],'model':newmodel})
                    stemcyls.append(Utils.makecylinder(model=newmodel,length=7,dense=60))
                    
        self.finalstems = finalstems
        self.stemcyls = stemcyls
        
    def Step_7_Ellipse_fit( self):
        for i in self.finalstems:
            if len(i['tree']) > 5:
                R = Utils.rotation_matrix_from_vectors(i['model'][3:6],[0,0,1])
                centeredtree = i['tree'] - i['model'][0:3]
                correctedcyl = (R @ centeredtree.T).T
                reg = LsqEllipse().fit(correctedcyl[:,0:2])
                center, a, b, phi = reg.as_parameters()
                Elipse_diameter = (3 * (a + b) - np.sqrt((3*a+b)*(a+3*b)))
                cylinder_diameter = i['model'][6]*2
                i['cylinder_diameter'] = cylinder_diameter
                i['Elipse_diameter'] = Elipse_diameter
                i['final_diameter'] = max(Elipse_diameter,cylinder_diameter)
            else:
                i['cylinder_diameter'] = None
                i['Elipse_diameter'] = None
                i['final_diameter'] = None
        
    def Full_Process(self, verticalityThresh = 0.06, NonNANcurvatureThresh = 0.1, tol=0.1, minc=40, maxc=6000000, max_angle = 0.4, lowstems_Height = 5, cutstems_Height = 5, searchRadius = 0.1):
        print('Step_1_Remove_Floor')
        self.Step_1_Remove_Floor()
        print('Step_2_Normal_Filtering')
        self.Step_2_Normal_Filtering(verticalityThresh, NonNANcurvatureThresh)
        print('Step_3_Eucladean_Clustering')
        self.Step_3_Eucladean_Clustering(tol, minc, maxc)
        print('Step_4_Group_Stems')
        self.Step_4_Group_Stems(max_angle)
        print('Step_5_Get_Ground_Level_Trees')
        self.Step_5_Get_Ground_Level_Trees(lowstems_Height, cutstems_Height)
        print('Step_6_Get_Cylinder_Tree_Models')
        self.Step_6_Get_Cylinder_Tree_Models(searchRadius)
        print('Step_7_Ellipse_fit')
        self.Step_7_Ellipse_fit()
        print('Done')
        
    def save_results(self, savelocation = 'results/myresults.csv'):
        Tree_Model_Info = [i['model'] for i in self.finalstems]
        Tree_diameter_Info = [i['final_diameter'] for i in self.finalstems]

        data = {'X':[],'Y':[],'Z':[],'DBH':[]}
        for i,j in zip(Tree_Model_Info, Tree_diameter_Info):
            data['X'].append(i[0])
            data['Y'].append(i[1])
            data['Z'].append(i[2])
            data['DBH'].append(j)

        pd.DataFrame.from_dict(data).to_csv(savelocation)