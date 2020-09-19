import sys
sys.path.append('..')
from laspy.file import File
import pclpy
import pcl
import VelLoader
import ARUSTUFF
import Plane
import Corners
import cv2
import imp
import glob
import os
import numpy as np
import open3d
import pdal
import pandas as pd
import pclpy
from Plane import tic,toc



def FloorRemove(points, scalar=0.2, slope=0.2, threshold=0.45, window=16.0, RGB=False):
    if (type(points) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(points) == pclpy.pcl.PointCloud.PointXYZ):
        pclpy.pcl.io.savePLYFile('LIDARRF.ply',points, binary_mode = True)
        json = """
        [
            "LIDARRF.ply",
            {
                "type":"filters.smrf",
                "scalar":2.0,
                "slope":2.0,
                "threshold":0.4,
                "window":16.0
            }
        ]
        """
    else:
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(points)
        open3d.write_point_cloud('LIDARRF.ply',pcd)
        json = """
        [
            "LIDARRF.ply",
            {
                "type":"filters.smrf",
                "scalar":0.2,
                "slope":0.2,
                "threshold":0.45,
                "window":16.0
            }
        ]
        """
    pipeline = pdal.Pipeline(json)
    pipeline.validate()
    pipeline.execute()
    arrays = pipeline.arrays
    points1 = arrays[0][arrays[0]['Classification']==1]
    points2 = arrays[0][arrays[0]['Classification']==2]

    Nogroundpoints = np.array(points1[['X','Y','Z']].tolist())
    ground = np.array(points2[['X','Y','Z']].tolist())
    if type(points) == pclpy.pcl.PointCloud.PointXYZRGB:
        if len(Nogroundpoints) > 0:
            Nogroundpoints = pclpy.pcl.PointCloud.PointXYZRGB(np.array(points1[['X','Y','Z']].tolist()),np.array(points1[['Red','Green','Blue']].tolist()))
        if len(ground) > 0:
            ground = pclpy.pcl.PointCloud.PointXYZRGB(np.array(points2[['X','Y','Z']].tolist()),np.array(points2[['Red','Green','Blue']].tolist()))
    return Nogroundpoints,ground

def RadiusOutlierRemoval(points , MinN=6, Radius=0.4, Organized=True):
    ROR = pclpy.pcl.filters.RadiusOutlierRemoval.PointXYZ()
    cloud = pclpy.pcl.PointCloud.PointXYZ(points)
    ROR.setInputCloud(cloud)
    ROR.setMinNeighborsInRadius(MinN)
    ROR.setRadiusSearch(Radius)
    ROR.setKeepOrganized(Organized)
    FilteredROR = pclpy.pcl.PointCloud.PointXYZ()
    ROR.filter(FilteredROR)
    return FilteredROR.xyz

def ExtractNormals(points, Ksearch = 0.1):
    cloud = pclpy.pcl.PointCloud.PointXYZ(points)
    segcloudNor = pclpy.pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    tree = pclpy.pcl.search.KdTree.PointXYZ()
    segcloudNor.setInputCloud(cloud)
    segcloudNor.setSearchMethod(tree)
    segcloudNor.setRadiusSearch(Ksearch)
    normals = pclpy.pcl.PointCloud.Normal()
    segcloudNor.compute(normals)
    return normals


def EucladeanClusterExtract(points, tol=2, minc=20, maxc=25000):
    filtered_points = pcl.PointCloud(points.astype(np.float32))
    tree = filtered_points.make_kdtree()
    ec = filtered_points.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tol)
    ec.set_MinClusterSize(minc)
    ec.set_MaxClusterSize(maxc)
    ec.set_SearchMethod (tree)
    cluster_indices = ec.Extract()
    cluster_list = []
    for indices in cluster_indices:
        points = filtered_points.to_array()[indices]
        cluster_list.append(points)
    return cluster_list

def RegionGrowing(Points, Ksearch=30, minc=20, maxc=100000, nn=30, smoothness=30.0, curvature=1.0):
    segcloud = pclpy.pcl.PointCloud.PointXYZ(Points)
    segcloudNor = pclpy.pcl.features.NormalEstimation.PointXYZ_Normal()
    tree = pclpy.pcl.search.KdTree.PointXYZ()

    segcloudNor.setInputCloud(segcloud)
    segcloudNor.setSearchMethod(tree)
    segcloudNor.setKSearch(Ksearch)
    normals = pclpy.pcl.PointCloud.Normal()
    segcloudNor.compute(normals)
    
    RGF = pclpy.pcl.segmentation.RegionGrowing.PointXYZ_Normal()
    RGF.setInputCloud(segcloud)
    RGF.setInputNormals(normals)
    RGF.setMinClusterSize(minc)
    RGF.setMaxClusterSize(maxc)
    RGF.setSearchMethod(tree)
    RGF.setNumberOfNeighbours(nn)
    RGF.setSmoothnessThreshold(smoothness / 180.0 * np.pi)
    RGF.setCurvatureThreshold(curvature)

    clusters = pclpy.pcl.vectors.PointIndices()
    RGF.extract(clusters)
    
    ppclusters = [segcloud.xyz[i2.indices] for i2 in clusters]
    return ppclusters

def segment_normals(points, searchRadius=20, model=pcl.SACMODEL_LINE, method=pcl.SAC_RANSAC, normalweight=0.0001, miter=1000, distance=0.5, rlim=[0,0.5]):
    
    segcloud = pcl.PointCloud(points)
    cylseg = segcloud.make_segmenter_normals(searchRadius=searchRadius)
    cylseg.set_optimize_coefficients(True)
    cylseg.set_model_type(model)
    cylseg.set_method_type(method)
    cylseg.set_normal_distance_weight(normalweight)
    cylseg.set_max_iterations(miter)
    cylseg.set_distance_threshold(distance)
    cylseg.set_radius_limits(rlim[0],rlim[1])
    indices, model = cylseg.segment()
    return indices, model




def findstemsLiDAR(pointsXYZ):
    Nogroundpoints,ground = FloorRemove(pointsXYZ)
    flatpoints = np.hstack([Nogroundpoints[:,0:2],np.zeros_like(Nogroundpoints)[:,0:1]])

    RRFpoints = RadiusOutlierRemoval(flatpoints)
    notgoodpoints = Nogroundpoints[np.isnan(RRFpoints[:,0])]
    goodpoints = Nogroundpoints[np.bitwise_not(np.isnan(RRFpoints[:,0]))]

    cluster_list = EucladeanClusterExtract(goodpoints)
    RGclusters = []
    for i in cluster_list:
        ppclusters = RegionGrowing(i)
        RGclusters.append(ppclusters)

    models = []
    stemsR = []
    for i in RGclusters:
        for p in i:
            indices, model = segment_normals(p)
            prop = len(p[indices])/len(p)
            if len(indices)>1 and prop>0. and np.arccos(np.dot([0,0,1],model[3:6]))<.6:
                points = p[indices]
                PC,_,_ = Plane.getPrincipalComponents(points)
                if PC[0]/PC[1]>10:
                    stemsR.append(points)
                    models.append(model)
    return stemsR,models

def voxelize(points,leaf = 0.1):
    if (type(points) == pclpy.pcl.PointCloud.PointXYZRGB):
        Cloud = points
        VF = pclpy.pcl.filters.VoxelGrid.PointXYZRGB()
        VFmm = pclpy.pcl.PointCloud.PointXYZRGB()
    else:
        Cloud = pclpy.pcl.PointCloud.PointXYZ(points)
        VF = pclpy.pcl.filters.VoxelGrid.PointXYZ()
        VFmm = pclpy.pcl.PointCloud.PointXYZ()
    
    VF.setLeafSize(leaf,leaf,leaf)
    VF.setInputCloud(Cloud)
    
    VF.filter(VFmm)
    if type(points) == pclpy.pcl.PointCloud.PointXYZRGB:
        return VFmm
    else:
        return VFmm.xyz