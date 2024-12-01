"""
MIT License

Copyright (c) 2021 porteratzo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import open3d as o3d
import numpy as np
import pdal
import os
import random


def voxelize(points, leaf=0.1):
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
    return_same = O3dPointsReturnSame(points)
    downpcd = return_same.get().voxel_down_sample(voxel_size=leaf)
    return return_same.get(downpcd)


class O3dPointsReturnSame:
    def __init__(self, points) -> None:
        """
        Initializes the O3dPointsReturnSame object.

        Args:
            points: np.ndarray | o3d.geometry.PointCloud
                The input points, either as a numpy array of shape (n, 3) or an
                open3d.geometry.PointCloud object.
        """
        if isinstance(points, o3d.geometry.PointCloud):
            self.is_cloud = True
            self.pointcloud = points
        else:
            self.is_cloud = False
            self.pointcloud = o3d.geometry.PointCloud()
            self.pointcloud.points = o3d.utility.Vector3dVector(points)

    def get(self, pointcloud=None):
        """
        Retrieves the current point cloud or points.

        Args:
            pointcloud: np.ndarray | o3d.geometry.PointCloud, optional
                The point cloud to retrieve. If None, uses the internally stored
                point cloud.

        Returns:
            np.ndarray | o3d.geometry.PointCloud: The point cloud or points.
            Returns the input type (numpy array or open3d.geometry.PointCloud).
        """
        if pointcloud is not None:
            cur_cloud = pointcloud
        else:
            cur_cloud = self.pointcloud
        if self.is_cloud:
            return cur_cloud
        else:
            return np.asarray(cur_cloud.points)

    def get_cloud(self):
        """
        Retrieves the internally stored point cloud.

        Returns:
            o3d.geometry.PointCloud: The internally stored point cloud.
        """
        return self.pointcloud

    def get_points(self):
        """
        Retrieves the points of the internally stored point cloud.

        Returns:
            np.ndarray: The points of the internally stored point cloud as a numpy array.
        """
        return np.asarray(self.pointcloud.points)


def floor_remove(
    points,
    set_max_window_size=20,
    set_slope=1.0,
    set_initial_distance=0.5,
    set_max_distance=3.0,
    cell_size=1,
):
    """
    Takes a point cloud and returns 2 pointclouds, the first for non ground points and the second
    for ground points

    Args:
        points : np.ndarray
            (n,3) point cloud

        set_max_window_size: int
            Set the maximum window size to be used in filtering ground returns.

        set_slope: float
            Set the slope value to be used in computing the height threshold.

        set_initial_distance: float
            Set the initial height above the parameterized ground surface to be considered a ground
            return.

        set_max_distance: float
            Set the maximum height above the parameterized ground surface to be considered a ground
            return.

    Returns:
        non_ground_points.xyz : np.narray (n,3)
            3d point cloud of only non ground points

        ground.xyz : np.narray (n,3)
            3d point cloud of only ground points

    """
    return_same = O3dPointsReturnSame(points)
    o3d.io.write_point_cloud("floorseg_temp_file.pcd", return_same.get())
    json = f"""
    [
        "floorseg_temp_file.pcd",
        {{
            "type":"filters.smrf",
            "cell":{cell_size},
            "scalar":{set_max_distance},
            "slope":{set_slope},
            "threshold":{set_initial_distance},
            "window":{set_max_window_size}
        }}
    ]
    """
    pipeline = pdal.Pipeline(json)
    pipeline.execute()
    arrays = pipeline.arrays
    points1 = arrays[0][arrays[0]["Classification"] == 1]
    points2 = arrays[0][arrays[0]["Classification"] == 2]
    Nogroundpoints = np.array(points1[["X", "Y", "Z"]].tolist())
    ground = np.array(points2[["X", "Y", "Z"]].tolist())
    os.remove("floorseg_temp_file.pcd")

    return return_same.get(Nogroundpoints), return_same.get(ground)


def radius_outlier_removal(points, min_n=6, radius=0.4, organized=True):
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

    return_same = O3dPointsReturnSame(points)
    ror_filter = return_same.get()
    cl, ind = ror_filter.remove_radius_outlier(nb_points=min_n, radius=radius)
    if organized:
        na_idx = np.delete(np.arange(len(ror_filter.points)), ind)
        return_points = np.asanyarray(ror_filter.points)
        return_points[na_idx] = np.nan
        _return_same = O3dPointsReturnSame(return_points)
        cl = return_same.get(_return_same.get())
    return cl


def compute_eigenvalues(chunk_of_matrices):
    _eigenvalues_list = []
    for matrix in chunk_of_matrices:
        h1, h2, h3 = np.linalg.eigvals(matrix)
        _eigenvalues_list.append(h3 / (h1 + h2 + h3))
    return _eigenvalues_list


def extract_normals(points, search_radius=0.1):
    """
    Takes a point cloud and approximates their normals using PCA

    Args:
        points : np.ndarray
            (n,3) point cloud

        search_radius: float
            Radius of the sphere a point can be in to be considered in the calculation of a sample
            points' normal

    Returns:
        normals : np.narray (n,3)
            (n,3) Normal vectors corresponding to the points in the input point cloud

    """
    return_same = O3dPointsReturnSame(points)
    PointCloudV = return_same.get_cloud()
    kd_tree = o3d.geometry.KDTreeSearchParamRadius(search_radius)
    PointCloudV.estimate_normals(kd_tree)
    PointCloudV.estimate_covariances(kd_tree)

    cov = np.asarray(PointCloudV.covariances)
    eigenvalues = np.linalg.eigvals(cov)
    result = eigenvalues[:, 0] / np.sum(eigenvalues, axis=1)

    return np.asarray(PointCloudV.normals), np.asarray(result)


def dbscan_cluster_extract(points, eps=2, min_points=20):
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
    return_same = O3dPointsReturnSame(points)
    PointCloudV = return_same.get_cloud()
    labels = PointCloudV.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    cluster_list = [
        return_same.get_points()[np.asarray(labels) == label]
        for label in set(labels)
        if label != -1
    ]
    return cluster_list


# Function to fit a cylinder model using RANSAC
def fit_cylinder_ransac(points, max_iterations=1000, distance_threshold=0.01, rlim=[None, None]):
    
    def compute_inliers(points, axis_point1, axis_point2, radius, distance_threshold):
        axis_vector = axis_point2 - axis_point1
        axis_length = np.linalg.norm(axis_vector)
        axis_unit_vector = axis_vector / axis_length

        point_vectors = points - axis_point1
        projection_lengths = np.dot(point_vectors, axis_unit_vector)
        projection_points = axis_point1 + np.outer(projection_lengths, axis_unit_vector)

        distances_to_axis = np.linalg.norm(projection_points - points, axis=1)
        surface_distances = np.abs(distances_to_axis - radius)

        return np.where(surface_distances < distance_threshold)[0]

    best_cylinder = None
    best_inliers = []

    num_points = points.shape[0]

    for _ in range(max_iterations):
        # Randomly sample 3 points to define a cylinder (axis and radius)
        sample_indices = random.sample(range(num_points), 3)
        sample_points = points[sample_indices]

        # Define the cylinder axis (from first two points) and radius (from third point)
        axis_point1, axis_point2, sample_point3 = sample_points
        axis_vector = axis_point2 - axis_point1
        radius = np.linalg.norm(
            np.cross(sample_point3 - axis_point1, axis_vector)
        ) / np.linalg.norm(axis_vector)

        # Measure inliers by calculating distances of all points to the cylinder

        if rlim[0] is None or radius > rlim[0]:
            if rlim[1] is None or radius < rlim[1]:
                inliers = compute_inliers(points, axis_point1, axis_point2, radius, distance_threshold)
                # Track the best model (with the most inliers)
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_cylinder = np.concatenate([axis_point1, axis_point2, np.array(radius)[None]])
    best_cylinder[3:6] = best_cylinder[3:6] - best_cylinder[0:3]
    best_cylinder[3:6] = best_cylinder[3:6]/np.linalg.norm(best_cylinder[3:6])
    return best_inliers, best_cylinder


# Function to fit a stick model using RANSAC
def fit_stick_ransac(point_cloud, max_iterations=1000, distance_threshold=0.01):
    def point_to_line_distance(point, line_point1, line_point2):
        # Vector from line_point1 to line_point2
        line_vector = line_point2 - line_point1
        point_vector = point - line_point1

        # Projection of point onto the line
        projection = np.dot(point_vector, line_vector) / np.linalg.norm(line_vector)
        projection_point = line_point1 + (projection / np.linalg.norm(line_vector)) * line_vector

        # Perpendicular distance from point to the line
        distance = np.linalg.norm(point - projection_point)

        return distance

    best_stick = None
    best_inliers = []

    points = np.asarray(point_cloud.points)
    num_points = points.shape[0]

    for _ in range(max_iterations):
        # Randomly sample 2 points to define a line (stick axis)
        sample_indices = random.sample(range(num_points), 2)
        sample_points = points[sample_indices]

        # Define the stick axis (line) using two points
        line_point1, line_point2 = sample_points

        # Measure inliers by calculating distances of all points to the stick axis (line)
        inliers = []
        for i, point in enumerate(points):
            dist = point_to_line_distance(point, line_point1, line_point2)
            if dist < distance_threshold:
                inliers.append(i)

        # Track the best model (with the most inliers)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_stick = (line_point1, line_point2)

    return best_stick, best_inliers


if False:

    def region_growing(
        Points,
        ksearch=30,
        minc=20,
        maxc=100000,
        nn=30,
        smoothness=30.0,
        curvature=1.0,
    ):
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
        non_ground_points, ground = floor_remove(pointsXYZ)
        flatpoints = np.hstack(
            [
                non_ground_points[:, 0:2],
                np.zeros_like(non_ground_points)[:, 0:1],
            ]
        )

        filtered_points = radius_outlier_removal(flatpoints)
        notgoodpoints = non_ground_points[np.isnan(filtered_points[:, 0])]
        goodpoints = non_ground_points[np.bitwise_not(np.isnan(filtered_points[:, 0]))]

        cluster_list = dbscan_cluster_extract(goodpoints)
        rg_clusters = []
        for i in cluster_list:
            rg_clusters.append(region_growing(i))

        models = []
        stem_clouds = []
        for i in rg_clusters:
            for p in i:
                indices, model = segment_normals(p)
                prop = len(p[indices]) / len(p)
                if (
                    len(indices) > 1
                    and prop > 0.0
                    and np.arccos(np.dot([0, 0, 1], model[3:6])) < 0.6
                ):
                    points = p[indices]
                    PC, _, _ = Plane.getPrincipalComponents(points)
                    if PC[0] / PC[1] > 10:
                        stem_clouds.append(points)
                        models.append(model)
        return stem_clouds, models
