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

import numpy as np
import pandas as pd
import treetool.seg_tree as seg_tree
import treetool.utils as utils
from ellipse import LsqEllipse
import os
import open3d as o3d
from typing import Union


def set_point_cloud(input_point_cloud: Union[np.ndarray, o3d.geometry.PointCloud]):
    """
    Resets the point cloud that treetool will process

    Args:
        point_cloud : np.narray
            The 3d point cloud of the forest that treetool will process, if it's a numpy array it
            should be shape (n,3)

    Returns:
        None
    """
    if input_point_cloud is not None:
        assert isinstance(input_point_cloud, o3d.geometry.PointCloud) or (
            type(input_point_cloud) is np.ndarray
        ), "Not valid point_cloud"
        if type(input_point_cloud) is np.ndarray:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(input_point_cloud)
        else:
            point_cloud = input_point_cloud
    return point_cloud


class treetool:
    """
    Our main class that holds all necessary methods to process a raw point into a list of all tree
      stem locations and DBHs
    """

    def __init__(self, point_cloud: Union[np.ndarray, o3d.geometry.PointCloud] = None):
        """
        Parameters
        ----------
        point_cloud : np.narray | o3d.geometry.PointCloud
            The 3d point cloud of the forest that treetool will process, if it's a numpy array it
              should be shape (n,3)
        """
        if point_cloud is not None:
            self.point_cloud = set_point_cloud(point_cloud)

    def step_1_remove_floor(
        self,
        set_max_window_size: int = 20,
        set_slope: float = 1.0,
        set_initial_distance: float = 0.5,
        set_max_distance: float = 3.0,
        set_cell_size: float = 1.0,
    ):
        """
        Applies ApproximateProgressiveMorphologicalFilter to point_cloud to separate the it's
        points into non_ground and ground points and assigns them to the non_ground_cloud and
        ground_cloud attributes

        Args:
            None

        Returns:
            None
        """
        no_ground_points, ground = seg_tree.floor_remove(
            self.point_cloud,
            set_max_window_size=set_max_window_size,
            set_slope=set_slope,
            set_initial_distance=set_initial_distance,
            set_max_distance=set_max_distance,
            cell_size=set_cell_size
        )
        self.non_ground_cloud: o3d.geometry.PointCloud = set_point_cloud(no_ground_points)
        self.ground_cloud: o3d.geometry.PointCloud = set_point_cloud(ground)

    def step_2_normal_filtering(
        self,
        search_radius=0.08,
        verticality_threshold=0.08,
        curvature_threshold=0.12,
        min_points=0,
    ):
        """
        Filters non_ground_cloud by approximating its normals and removing points with a high
        curvature and a non near horizontal normal
        the points that remained are assigned to

        Args:
            search_radius : float
                Maximum distance of the points to a sample point that will be used to approximate a
                the sample point's normal

            verticality_threshold: float
                Threshold in radians for filtering the verticality of each point, we determine
                obtaining the dot product of each points normal by a vertical vector [0,0,1]

            curvature_threshold: float
                Threshold [0-1] for filtering the curvature of each point, the curvature is given
                by lambda_0/(lambda_0 + lambda_1 + lambda_2) where lambda_j is the
                j-th eigenvalue of the covariance matrix of radius of points around each query
                point and lambda_0 < lambda_1 < lambda_2

        Returns:
            None
        """
        # get point normals
        if min_points > 0:
            subject_cloud = seg_tree.radius_outlier_removal(
                self.non_ground_cloud.points,
                min_points,
                search_radius,
                organized=False,
            )
        else:
            subject_cloud = self.non_ground_cloud.points
        non_ground_normals, non_ground_curvature = seg_tree.extract_normals(
            subject_cloud, search_radius
        )
        # remove Nan points
        non_nan_mask = np.bitwise_not(np.isnan(non_ground_normals[:, 0]))
        non_nan_cloud = np.asarray(subject_cloud)[non_nan_mask]
        non_nan_normals = non_ground_normals[non_nan_mask]
        non_nan_curvature = non_ground_curvature[non_nan_mask]

        # get mask by filtering verticality and curvature
        verticality = np.abs(np.dot(non_nan_normals, [[0], [0], [1]]))
        verticality_mask = (verticality < verticality_threshold) & (
            -verticality_threshold < verticality
        )
        curvature_mask = non_nan_curvature < curvature_threshold
        verticality_curvature_mask = verticality_mask.ravel() & curvature_mask.ravel()

        only_horizontal_points = non_nan_cloud[verticality_curvature_mask]
        only_horizontal_normals = non_nan_normals[verticality_curvature_mask]

        self.curvature = non_nan_curvature
        self.verticality = verticality

        # set filtered and non filtered points
        self.non_ground_normals = non_ground_normals
        self.non_filtered_normals = non_nan_normals
        self.non_filtered_points = non_nan_cloud
        self.filtered_points = only_horizontal_points
        self.filtered_normals = only_horizontal_normals

    def step_3_dbscan_clustering(self, eps=0.1, min_cluster_size=40):
        """
        Clusters filtered_points with euclidean clustering and assigns them to attribute
        cluster_list

        Args:
            tolerance : float
                Maximum distance a point can be from a cluster for that point to be included in the
                cluster.

            min_cluster_size: int
                Minimum number of points a cluster must have to not be discarded

            max_cluster_size: int
                Maximum number of points a cluster must have to not be discarded

        Returns:
            None
        """
        self.cluster_list = seg_tree.dbscan_cluster_extract(
            self.filtered_points,
            eps=eps,
            min_points=min_cluster_size,
        )

    def step_4_group_stems(self, max_distance=0.4):
        """
        For each cluster in attribute cluster_list, test if its centroid is near the line formed by
        the first principal vector of another cluster parting from the centroid of that cluster
        and if so, join the two clusters

        Args:
            max_distance : float
                Maximum distance a point can be from the line formed by the first principal vector
                of another cluster parting from the centroid of that cluster

        Returns:
            None
        """
        # Get required info from clusters
        stem_groups = []
        for n, p in enumerate(self.cluster_list):
            Centroid = np.mean(p, axis=0)
            vT, S = utils.getPrincipalVectors(p - Centroid)
            straightness = S[0] / (S[0] + S[1] + S[2])

            clusters_dict = {}
            clusters_dict["cloud"] = p
            clusters_dict["straightness"] = straightness
            clusters_dict["center"] = Centroid
            clusters_dict["direction"] = vT
            stem_groups.append(clusters_dict)

        # For each cluster, test if its centroid is near the line formed by the first principal
        # vector of another cluster parting from the centroid of that cluster
        # if so, join the two clusters
        temp_stems = [i["cloud"] for i in stem_groups]
        for treenumber1 in reversed(range(0, len(temp_stems))):
            for treenumber2 in reversed(range(0, treenumber1)):
                center1 = stem_groups[treenumber1]["center"]
                center2 = stem_groups[treenumber2]["center"]
                if np.linalg.norm(center1[:2] - center2[:2]) < 2:
                    vector1 = stem_groups[treenumber1]["direction"][0]
                    vector2 = stem_groups[treenumber2]["direction"][0]
                    dist1 = utils.DistPoint2Line(center2, vector1 + center1, center1)
                    dist2 = utils.DistPoint2Line(center1, vector2 + center2, center2)
                    if (dist1 < max_distance) | (dist2 < max_distance):
                        temp_stems[treenumber2] = np.vstack(
                            [
                                temp_stems[treenumber2],
                                temp_stems.pop(treenumber1),
                            ]
                        )
                        break

        self.complete_Stems = temp_stems
        self.stem_groups = stem_groups

    def step_5_get_ground_level_trees(
        self,
        lowstems_height=5,
        cutstems_height=5,
        use_sampling=False,
        dont_cut=False,
    ):
        """
        Filters stems to only keep those near the ground and crops them up to a certain height

        Args:
            lowstems_height: int
                Minimum number of points a cluster must have to not be discarded

            cutstems_height: int
                Maximum number of points a cluster must have to not be discarded

        Returns:
            None
        """
        # Generate a bivariate quadratic equation to model the ground
        ground_points = np.asarray(self.ground_cloud.points)
        if not use_sampling:
            A = np.c_[
                np.ones(ground_points.shape[0]),
                ground_points[:, :2],
                np.prod(ground_points[:, :2], axis=1),
                ground_points[:, :2] ** 2,
            ]
            self.ground_model_c, _, _, _ = np.linalg.lstsq(A, ground_points[:, 2], rcond=None)

        # Obtain a ground point for each stem by taking the XY component of the centroid
        # and obtaining the coresponding Z coordinate from our quadratic ground model
        self.stems_with_ground = []
        for i in self.complete_Stems:
            center = np.mean(i, 0)
            X, Y = center[:2]
            if not use_sampling:
                Z = np.dot(
                    np.c_[np.ones(X.shape), X, Y, X * Y, X**2, Y**2],
                    self.ground_model_c,
                )
            else:
                _size = 0.5
                while True:
                    sub_pcd = o3d.geometry.crop_point_cloud(
                        self.ground_cloud,
                        np.hstack([X - _size, Y - _size, -100, 1]),
                        np.hstack([X + _size, Y + _size, 100, 1]),
                    )
                    if len(np.asarray(sub_pcd.points)) > 5:
                        Z = [np.mean(np.asarray(sub_pcd.points)[:, 2])]
                        break
                    _size += 0.25

            self.stems_with_ground.append([i, [X, Y, Z[0]]])

        # Filter stems that do not have points below our lowstems_height threshold
        low_stems = [
            i
            for i in self.stems_with_ground
            if np.min(i[0], axis=0)[2] < (lowstems_height + i[1][2])
        ]
        # Crop points above cutstems_height threshold
        if not dont_cut:
            cut_stems = [[i[0][i[0][:, 2] < (cutstems_height + i[1][2])], i[1]] for i in low_stems]
        else:
            cut_stems = low_stems

        self.cut_stems = cut_stems
        self.low_stems = [i[0] for i in cut_stems]

    def step_6_get_cylinder_tree_models(self, search_radius=0.1, distance=0.08, stick=False):
        """
        For each cut stem we use ransac to extract a cylinder model

        Args:
            search_radius : float
                Maximum distance of the points to a sample point that will be used to approximate a
                the sample point's normal

        Returns:
            None
        """
        final_stems = []
        visualization_cylinders = []
        for p in self.cut_stems:
            # Segment to cylinders
            stem_points = p[0]
            if len(stem_points) <= 1:
                continue
            if stick:
                indices, model = seg_tree.fit_stick_ransac(
                    stem_points, max_iterations=1000, distance_threshold=0.4
                )
            else:
                indices, model = seg_tree.fit_cylinder_ransac(
                    stem_points, max_iterations=1000, distance_threshold=0.08, rlim=[0, 0.4]
                )
            # If the model has more than 10 points
            if len(indices) > 10:
                # If the model finds an upright cylinder
                if abs(np.dot(model[3:6], [0, 0, 1]) / np.linalg.norm(model[3:6])) > 0.5:
                    # Get centroid
                    model = np.array(model)
                    Z = 1.3 + p[1][2]
                    Y = model[1] + model[4] * (Z - model[2]) / model[5]
                    X = model[0] + model[3] * (Z - model[2]) / model[5]
                    model[0:3] = np.array([X, Y, Z])
                    # make sure the vector is pointing upward
                    model[3:6] = utils.similarize(model[3:6], [0, 0, 1])
                    final_stems.append(
                        {
                            "tree": stem_points[indices],
                            "model": model,
                            "ground": p[1][2],
                        }
                    )
                    visualization_cylinders.append(
                        utils.makecylinder(model=model, height=7, density=60)
                    )

        self.finalstems = final_stems
        self.visualization_cylinders = visualization_cylinders

    def step_7_ellipse_fit(self, height_ll=-1, height_ul=-1):
        """
        Extract the cylinder and ellipse diameter of each stem

        Args:
            None

        Returns:
            None
        """
        for i in self.finalstems:
            # if the tree points has enough points to fit a ellipse
            if len(i["tree"]) > 5:
                # find a matrix that rotates the stem to be colinear to the z axis
                R = utils.rotation_matrix_from_vectors(i["model"][3:6], [0, 0, 1])
                # we center the stem to the origen then rotate it
                centeredtree = i["tree"] - i["model"][0:3]
                correctedcyl = (R @ centeredtree.T).T
                # fit an ellipse using only the xy coordinates
                try:
                    if height_ll != -1:
                        correctedcyl = correctedcyl[:, 2] > height_ll
                    if height_ul != -1:
                        correctedcyl = correctedcyl[:, 2] < height_ul
                    reg = LsqEllipse().fit(correctedcyl[:, 0:2])
                    center, a, b, phi = reg.as_parameters()

                    ellipse_diameter = 3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))
                except np.linalg.LinAlgError:
                    ellipse_diameter = i["model"][6] * 2
                except IndexError:
                    ellipse_diameter = i["model"][6] * 2
                cylinder_diameter = i["model"][6] * 2
                i["cylinder_diameter"] = cylinder_diameter
                i["ellipse_diameter"] = ellipse_diameter
                i["final_diameter"] = max(ellipse_diameter, cylinder_diameter)
                n_model = i["model"]
                n_model[6] = i["final_diameter"]
                i["vis_cyl"] = utils.makecylinder(model=n_model, height=7, density=60)
            else:
                i["cylinder_diameter"] = None
                i["ellipse_diameter"] = None
                i["final_diameter"] = None
                i["vis_cyl"] = None

    def full_process(
        self,
        search_radius=0.1,
        verticality_threshold=0.06,
        curvature_threshold=0.1,
        tolerance=0.1,
        min_cluster_size=40,
        max_cluster_size=6000000,
        max_distance=0.4,
        lowstems_height=5,
        cutstems_height=5,
        searchRadius_cylinder=0.1,
    ):
        """
        Clusters filtered_points with euclidean clustering and assigns them to attribute
        cluster_list

        Args:
            search_radius : float
                Maximum distance of the points to a sample point that will be used to approximate a
                the sample point's normal

            verticality_threshold: float
                Threshold in radians for filtering the verticality of each point, we determine
                obtaining the dot product of each points normal by a vertical vector [0,0,1]

            curvature_threshold: float
                Threshold [0-1] for filtering the curvature of each point, the curvature is given
                by lambda_0/(lambda_0 + lambda_1 + lambda_2) where lambda_j is the
                j-th eigenvalue of the covariance matrix of radius of points around each query
                point and lambda_0 < lambda_1 < lambda_2

            tolerance : float
                Maximum distance a point can be from a cluster for that point to be included in the
                cluster.

            min_cluster_size: int
                Minimum number of points a cluster must have to not be discarded

            max_cluster_size: int
                Maximum number of points a cluster must have to not be discarded

            max_distance : float
                Maximum distance a point can be from the line formed by the first principal vector
                of another cluster parting from the centroid of that cluster

            lowstems_height: int
                Minimum number of points a cluster must have to not be discarded

            cutstems_height: int
                Maximum number of points a cluster must have to not be discarded

            searchRadius_cylinder : float
                Maximum distance of the points to a sample point that will be used to approximate a
                the sample point's normal


        Returns:
            None
                minimum number of points a cluster must have to not be discarded

        """
        print("step_1_Remove_Floor")
        self.step_1_remove_floor()
        print("step_2_normal_filtering")
        self.step_2_normal_filtering(search_radius, verticality_threshold, curvature_threshold)
        print("step_3_euclidean_clustering")
        self.step_3_dbscan_clustering(tolerance, min_cluster_size)
        print("step_4_Group_Stems")
        self.step_4_group_stems(max_distance)
        print("step_5_Get_Ground_Level_Trees")
        self.step_5_get_ground_level_trees(lowstems_height, cutstems_height)
        print("step_6_Get_Cylinder_Tree_Models")
        self.step_6_get_cylinder_tree_models(searchRadius_cylinder)
        print("step_7_Ellipse_fit")
        self.step_7_ellipse_fit()
        print("Done")

    def save_results(self, save_location="results/myresults.csv"):
        """
        Save a csv with XYZ and DBH of each detected tree

        Args:
            savelocation : str
                path to save file

        Returns:
            None
        """
        tree_model_info = [i["model"] for i in self.finalstems]
        tree_diameter_info = [i["final_diameter"] for i in self.finalstems]

        data = {"X": [], "Y": [], "Z": [], "DBH": []}
        for i, j in zip(tree_model_info, tree_diameter_info):
            data["X"].append(i[0])
            data["Y"].append(i[1])
            data["Z"].append(i[2])
            data["DBH"].append(j)

        os.makedirs(os.path.dirname(save_location), exist_ok=True)

        pd.DataFrame.from_dict(data).to_csv(save_location)
