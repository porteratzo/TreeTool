import pclpy
import numpy as np

def floor_remove(points, set_max_window_size = 20, set_slope = 1.0, set_initial_distance = 0.5, set_max_distance = 3.0):
    """
        Takes a point cloud and returns 2 pointclouds, the first for non ground points and the second for ground points

        Args:
            points : np.ndarray
                (n,3) point cloud
            
            set_max_window_size: int
                Set the maximum window size to be used in filtering ground returns. 

            set_slope: float
                Set the slope value to be used in computing the height threshold. 

            set_initial_distance: float
                Set the initial height above the parameterized ground surface to be considered a ground return. 

            set_max_distance: float
                Set the maximum height above the parameterized ground surface to be considered a ground return. 

        Returns:
            non_ground_points.xyz : np.narray (n,3)
                3d point cloud of only non ground points

            ground.xyz : np.narray (n,3)
                3d point cloud of only ground points
                
        """
    PointCloud = points
    ind = pclpy.pcl.vectors.Int()
    pmf = pclpy.pcl.segmentation.ApproximateProgressiveMorphologicalFilter.PointXYZ()
    pmf.setInputCloud(PointCloud)
    pmf.setMaxWindowSize(set_max_window_size)
    pmf.setSlope(set_slope)
    pmf.setInitialDistance(set_initial_distance)
    pmf.setMaxDistance(set_max_distance)
    pmf.extract(ind)
    ext = pclpy.pcl.filters.ExtractIndices.PointXYZ()
    ground = pclpy.pcl.PointCloud.PointXYZ()
    non_ground_points = pclpy.pcl.PointCloud.PointXYZ()
    ext.setInputCloud(PointCloud)
    ext.setIndices(ind)
    ext.filter(ground)
    ext.setNegative(True)
    ext.filter(non_ground_points)

    return non_ground_points.xyz, ground.xyz

def radius_outlier_removal(points , min_n=6, radius=0.4, organized=True):
    """
        Takes a point cloud and removes points that have less than minn neigbors in a certain radius

        Args:
            points : np.ndarray
                (n,3) point cloud
            
            min_n: int
                Neighbor threshold to keep a point

            radius: float
                Radius of the sphere a point can be in to be considered a neighbor of our sample point

            organized: bool
                If true outlier points are set to nan instead of removing the points from the cloud
                

        Returns:
            filtered_point_cloud.xyz : np.narray (n,3)
                (n,3) Pointcloud with outliers removed
                
        """
    ror_filter = pclpy.pcl.filters.RadiusOutlierRemoval.PointXYZ()
    cloud = pclpy.pcl.PointCloud.PointXYZ(points)
    ror_filter.setInputCloud(cloud)
    ror_filter.setMinNeighborsInRadius(min_n)
    ror_filter.setRadiusSearch(radius)
    ror_filter.setKeepOrganized(organized)
    filtered_point_cloud = pclpy.pcl.PointCloud.PointXYZ()
    ror_filter.filter(filtered_point_cloud)
    return filtered_point_cloud.xyz

def extract_normals(points, search_radius = 0.1):
    """
        Takes a point cloud and approximates their normals using PCA

        Args:
            points : np.ndarray
                (n,3) point cloud
            
            search_radius: float
                Radius of the sphere a point can be in to be considered in the calculation of a sample points' normal

        Returns:
            normals : np.narray (n,3)
                (n,3) Normal vectors corresponding to the points in the input point cloud
                
        """
    cloud = pclpy.pcl.PointCloud.PointXYZ(points)
    cloud_normal_estimator = pclpy.pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    tree = pclpy.pcl.search.KdTree.PointXYZ()
    cloud_normal_estimator.setInputCloud(cloud)
    cloud_normal_estimator.setSearchMethod(tree)
    cloud_normal_estimator.setRadiusSearch(search_radius)
    normals = pclpy.pcl.PointCloud.Normal()
    cloud_normal_estimator.compute(normals)
    return normals


def euclidean_cluster_extract(points, tolerance=2, min_cluster_size=20, max_cluster_size=25000):
    """
        Takes a point cloud and clusters the points with euclidean clustering

        Args:
            points : np.ndarray
                (n,3) point cloud
            
            tolerance: int
                Maximum distance a point can be to a cluster to added to that cluster

            min_cluster_size: int
                Minimum number of points a cluster must have to be returned

            max_cluster_size: int
                Maximum number of points a cluster must have to be returned
                

        Returns:
            cluster_list : list
                List of (n,3) Pointclouds representing each cluster
                
        """
    filtered_points = pclpy.pcl.segmentation.EuclideanClusterExtraction.PointXYZ()
    kd_tree = pclpy.pcl.search.KdTree.PointXYZ()
    points_to_cluster = pclpy.pcl.PointCloud.PointXYZ(points)
    
    kd_tree.setInputCloud(points_to_cluster)
    filtered_points.setInputCloud(points_to_cluster)
    filtered_points.setClusterTolerance(tolerance)
    filtered_points.setMinClusterSize(min_cluster_size)
    filtered_points.setMaxClusterSize(max_cluster_size)
    filtered_points.setSearchMethod(kd_tree)

    point_indexes = pclpy.pcl.vectors.PointIndices()
    filtered_points.extract(point_indexes)

    cluster_list = [points_to_cluster.xyz[i2.indices] for i2 in point_indexes]
    return cluster_list

def region_growing(Points, ksearch=30, minc=20, maxc=100000, nn=30, smoothness=30.0, curvature=1.0):
    """
        Takes a point cloud and clusters the points with region growing

        Args:
            points : np.ndarray
                (n,3) point cloud
            
            Ksearch: int
                Number of points used to estimate a points normal

            minc: int
                Minimum number of points a cluster must have to be returned

            maxc: int
                Maximum number of points a cluster must have to be returned
                
            nn: int
                Number of nearest neighbors used by the region growing algorithm

            smoothness:
                Smoothness threshold used in region growing

            curvature:
                Curvature threshold used in region growing

        Returns:
            region_growing_clusters: list
                list of (n,3) Pointclouds representing each cluster
                
        """
    pointcloud = pclpy.pcl.PointCloud.PointXYZ(Points)
    pointcloud_normals = pclpy.pcl.features.NormalEstimation.PointXYZ_Normal()
    tree = pclpy.pcl.search.KdTree.PointXYZ()

    pointcloud_normals.setInputCloud(pointcloud)
    pointcloud_normals.setSearchMethod(tree)
    pointcloud_normals.setKSearch(ksearch)
    normals = pclpy.pcl.PointCloud.Normal()
    pointcloud_normals.compute(normals)
    
    region_growing_clusterer = pclpy.pcl.segmentation.RegionGrowing.PointXYZ_Normal()
    region_growing_clusterer.setInputCloud(pointcloud)
    region_growing_clusterer.setInputNormals(normals)
    region_growing_clusterer.setMinClusterSize(minc)
    region_growing_clusterer.setMaxClusterSize(maxc)
    region_growing_clusterer.setSearchMethod(tree)
    region_growing_clusterer.setNumberOfNeighbours(nn)
    region_growing_clusterer.setSmoothnessThreshold(smoothness / 180.0 * np.pi)
    region_growing_clusterer.setCurvatureThreshold(curvature)

    clusters = pclpy.pcl.vectors.PointIndices()
    region_growing_clusterer.extract(clusters)
    
    region_growing_clusters = [pointcloud.xyz[i2.indices] for i2 in clusters]
    return region_growing_clusters


def segment(points, model=pclpy.pcl.sample_consensus.SACMODEL_LINE, method=pclpy.pcl.sample_consensus.SAC_RANSAC, miter=1000, distance=0.5, rlim=[0,0.5]):   
    """
        Takes a point cloud and removes points that have less than minn neigbors in a certain radius

        Args:
            points : np.ndarray
                (n,3) point cloud
            
            model: int
                A pclpy.pcl.sample_consensus.MODEL value representing a ransac model

            method: float
                pclpy.pcl.sample_consensus.METHOD to use

            miter: bool
                Maximum iterations for ransac

            distance:
                Maximum distance a point can be from the model

            rlim:
                Radius limit for cylinder model
                

        Returns:
            pI.indices: np.narray (n)
                Indices of points that fit the model

            Mc.values: np.narray (n)
                Model coefficients
                
        """
    pointcloud = pclpy.pcl.PointCloud.PointXYZ(points)
    segmenter = pclpy.pcl.segmentation.SACSegmentation.PointXYZ()

    segmenter.setInputCloud(pointcloud)
    segmenter.setDistanceThreshold(distance)
    segmenter.setOptimizeCoefficients(True)
    segmenter.setMethodType(method)
    segmenter.setModelType(model)
    segmenter.setMaxIterations(miter)
    segmenter.setRadiusLimits(rlim[0],rlim[1])
    pI = pclpy.pcl.PointIndices()
    Mc = pclpy.pcl.ModelCoefficients()
    segmenter.segment(pI,Mc)
    return pI.indices, Mc.values

def segment_normals(points, search_radius=20, model=pclpy.pcl.sample_consensus.SACMODEL_LINE, method=pclpy.pcl.sample_consensus.SAC_RANSAC, normalweight=0.0001, miter=1000, distance=0.5, rlim=[0,0.5]):
    """
        Takes a point cloud and removes points that have less than minn neigbors in a certain radius

        Args:
            points : np.ndarray
                (n,3) point cloud

            search_radius: float
                Radius of the sphere a point can be in to be considered in the calculation of a sample points' normal
            
            model: int
                A pclpy.pcl.sample_consensus.MODEL value representing a ransac model

            method: float
                pclpy.pcl.sample_consensus.METHOD to use

            normalweight:
                Normal weight for ransacfromnormals

            miter: bool
                Maximum iterations for ransac

            distance:
                Maximum distance a point can be from the model

            rlim:
                Radius limit for cylinder model
                

        Returns:
            pI.indices: np.narray (n)
                Indices of points that fit the model

            Mc.values: np.narray (n)
                Model coefficients
                
        """
    pointcloud_normals = extract_normals(points, search_radius)
    
    pointcloud = pclpy.pcl.PointCloud.PointXYZ(points)
    segmenter = pclpy.pcl.segmentation.SACSegmentationFromNormals.PointXYZ_Normal()

    segmenter.setInputCloud(pointcloud)
    segmenter.setInputNormals(pointcloud_normals)
    segmenter.setDistanceThreshold(distance)
    segmenter.setOptimizeCoefficients(True)
    segmenter.setMethodType(method)
    segmenter.setModelType(model)
    segmenter.setMaxIterations(miter)
    segmenter.setRadiusLimits(rlim[0],rlim[1])
    segmenter.setNormalDistanceWeight(normalweight)
    pI = pclpy.pcl.PointIndices()
    Mc = pclpy.pcl.ModelCoefficients()
    segmenter.segment(pI,Mc)
    return pI.indices, Mc.values


def findstemsLiDAR(pointsXYZ):
    """
        Takes a point cloud from a Cylindrical LiDAR and extract stems and their models

        Args:
            points : np.ndarray
                (n,3) point cloud                

        Returns:
            stemsR : list(np.narray (n,3))
                List of (n,3) Pointclouds belonging to each stem

            models : list(np.narray (n))
                List of model coefficients corresponding to each extracted stem
                
        """
    non_ground_points,ground = floor_remove(pointsXYZ)
    flatpoints = np.hstack([non_ground_points[:,0:2],np.zeros_like(non_ground_points)[:,0:1]])

    filtered_points = radius_outlier_removal(flatpoints)
    notgoodpoints = non_ground_points[np.isnan(filtered_points[:,0])]
    goodpoints = non_ground_points[np.bitwise_not(np.isnan(filtered_points[:,0]))]

    cluster_list = euclidean_cluster_extract(goodpoints)
    rg_clusters = []
    for i in cluster_list:
        rg_clusters.append(region_growing(i))

    models = []
    stem_clouds = []
    for i in rg_clusters:
        for p in i:
            indices, model = segment_normals(p)
            prop = len(p[indices])/len(p)
            if len(indices)>1 and prop>0. and np.arccos(np.dot([0,0,1],model[3:6]))<.6:
                points = p[indices]
                PC,_,_ = Plane.getPrincipalComponents(points)
                if PC[0]/PC[1]>10:
                    stem_clouds.append(points)
                    models.append(model)
    return stem_clouds,models

def voxelize(points,leaf = 0.1):
    """
        Use voxelgrid to subsample a pointcloud

        Args:
            points : np.ndarray
                (n,3) point cloud
            
            leaf: float
                Voxelsize

        Returns:
            VFmm: np.narray (n,3)
                (n,3) subsampled Pointcloud
                
        """
    if (type(points) == pclpy.pcl.PointCloud.PointXYZRGB):
        cloud = points
        voxel_filter = pclpy.pcl.filters.VoxelGrid.PointXYZRGB()
        filtered_pointcloud = pclpy.pcl.PointCloud.PointXYZRGB()
    else:
        cloud = pclpy.pcl.PointCloud.PointXYZ(points)
        voxel_filter = pclpy.pcl.filters.VoxelGrid.PointXYZ()
        filtered_pointcloud = pclpy.pcl.PointCloud.PointXYZ()
    
    voxel_filter.setLeafSize(leaf,leaf,leaf)
    voxel_filter.setInputCloud(cloud)
    
    voxel_filter.filter(filtered_pointcloud)
    if type(points) == pclpy.pcl.PointCloud.PointXYZRGB:
        return filtered_pointcloud
    else:
        return filtered_pointcloud.xyz
    
