import sys
import pclpy
import numpy as np
import pdal
import pandas as pd
import os
import Utils
import open3d


def FloorRemove(points, scalar=0.2, slope=0.2, threshold=0.45, window=16.0, RGB=False):
    #pclpy.pcl.io.savePLYFile('LIDARRF.ply',points, binary_mode = True)
    plycloud = Utils.convertcloud(points.xyz)
    open3d.io.write_point_cloud('LIDARRF.ply',plycloud)
    json = """
    [
        "LIDARRF.ply",
        {
            "type":"filters.smrf",
            "scalar":1.25,
            "slope":0.15,
            "threshold":0.5,
            "window":18.0
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
    os.remove('LIDARRF.ply')
    
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
    filtered_points = pclpy.pcl.segmentation.EuclideanClusterExtraction.PointXYZ()
    kdtree = pclpy.pcl.search.KdTree.PointXYZ()
    pointstocluster = pclpy.pcl.PointCloud.PointXYZ(points)
    
    kdtree.setInputCloud(pointstocluster)
    filtered_points.setInputCloud(pointstocluster)
    filtered_points.setClusterTolerance(tol)
    filtered_points.setMinClusterSize(minc)
    filtered_points.setMaxClusterSize(maxc)
    filtered_points.setSearchMethod(kdtree)

    pI = pclpy.pcl.vectors.PointIndices()
    filtered_points.extract(pI)

    cluster_list = [pointstocluster.xyz[i2.indices] for i2 in pI]
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


def segment(points, model=pclpy.pcl.sample_consensus.SACMODEL_LINE, method=pclpy.pcl.sample_consensus.SAC_RANSAC, miter=1000, distance=0.5, rlim=[0,0.5]):   
    segcloud = pclpy.pcl.PointCloud.PointXYZ(points)
    cylseg = pclpy.pcl.segmentation.SACSegmentation.PointXYZ()

    cylseg.setInputCloud(segcloud)
    cylseg.setDistanceThreshold(distance)
    cylseg.setOptimizeCoefficients(True)
    cylseg.setMethodType(method)
    cylseg.setModelType(model)
    cylseg.setMaxIterations(miter)
    cylseg.setRadiusLimits(rlim[0],rlim[1])
    pI = pclpy.pcl.PointIndices()
    Mc = pclpy.pcl.ModelCoefficients()
    cylseg.segment(pI,Mc)
    return pI.indices, Mc.values

def segment_normals(points, searchRadius=20, model=pclpy.pcl.sample_consensus.SACMODEL_LINE, method=pclpy.pcl.sample_consensus.SAC_RANSAC, normalweight=0.0001, miter=1000, distance=0.5, rlim=[0,0.5]):
    segNormals = ExtractNormals(points, searchRadius)
    
    segcloud = pclpy.pcl.PointCloud.PointXYZ(points)
    cylseg = pclpy.pcl.segmentation.SACSegmentationFromNormals.PointXYZ_Normal()

    cylseg.setInputCloud(segcloud)
    cylseg.setInputNormals(segNormals)
    cylseg.setDistanceThreshold(distance)
    cylseg.setOptimizeCoefficients(True)
    cylseg.setMethodType(method)
    cylseg.setModelType(model)
    cylseg.setMaxIterations(miter)
    cylseg.setRadiusLimits(rlim[0],rlim[1])
    cylseg.setNormalDistanceWeight(normalweight)
    pI = pclpy.pcl.PointIndices()
    Mc = pclpy.pcl.ModelCoefficients()
    cylseg.segment(pI,Mc)
    return pI.indices, Mc.values

"""
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
"""



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
    
