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
import pclpy
from matplotlib import cm
import matplotlib.pyplot as plt
import open3d
import libraries.seg_tree as seg_tree

def rotation_matrix_from_vectors(vector1, vector2):
    """
        Finds a rotation matrix that can rotate vector1 to align with vector 2

        Args:
            vector1: np.narray (3)
                Vector we would apply the rotation to
        
            vector2: np.narray (3)
                Vector that will be aligned to

        Returns:
            rotation_matrix: np.narray (3,3)
                Rotation matrix that when applied to vector1 will turn it to the same direction as vector2
        """
    if all(np.abs(vector1)==np.abs(vector2)):
        return np.eye(3)
    a, b = (vector1 / np.linalg.norm(vector1)).reshape(3), (vector2 / np.linalg.norm(vector2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + matrix + matrix.dot(matrix) * ((1 - c) / (s ** 2))
    return rotation_matrix

def angle_between_vectors(vector1,vector2):
    """
        Finds the angle between 2 vectors

        Args:
            vec1: np.narray (3)
                First vector to measure angle from
        
            vec2: np.narray (3)
                Second vector to measure angle to

        Returns:
            None
        """
    value = np.sum(np.multiply(vector1, vector2)) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    if (value<-1) | (value>1):
        value = np.sign(value)
    angle = np.arccos(value)
    return angle

def makecylinder(model=[0,0,0,1,0,0,1],height = 1,density=10):
    """
        Makes a point cloud of a cylinder given a (7) parameter cylinder model and a length and density

        Args:
            model: np.narray (7)
                7 parameter cylinder model

            height: float
                Desired height of the generated cylinder

            density: int
                Desired density of the generated cylinder, 
                this density is determines the amount of points on each ring that composes the cylinder and on how many rings the cylinder will have

        Returns:
            rotated_cylinder: np.narray (n,3)
                3d point cloud of the desired cylinder
        """
    # extract info from cylinder model
    radius = model[6]
    X,Y,Z = model[:3]
    # get 3d points to make an upright cylinder centered to the origin
    n = np.arange(0,360,int(360/density))
    height = np.arange(0,height,height/density)
    n = np.deg2rad(n)
    x,z = np.meshgrid(n,height)
    x = x.flatten()
    z = z.flatten()
    cyl = np.vstack([np.cos(x)*radius,np.sin(x)*radius,z]).T
    # rotate and translate the cylinder to fit the model
    rotation = rotation_matrix_from_vectors([0,0,1],model[3:6])
    rotated_cylinder = np.matmul(rotation,cyl.T).T + np.array([X,Y,Z])
    return rotated_cylinder   

def DistPoint2Line(point,line_point1, line_point2=np.array([0,0,0])):
    """
        Get minimum distance from a point to a line composed by 2 points

        Args:
            point: np.narray (3)
                XYZ coordinates of the 3d point
            
            line_point1: np.narray (3)
                XYZ coordinates of the first 3d point that composes the line if line_point2 is not given, line_point2 defaults to 0,0,0

            line_point2: np.narray (3)
                XYZ coordinates of the second 3d point that composes the line, if not given defaults to 0,0,0

        Returns:
            distance: float
                Shortest distance from point to the line composed by line_point1 line_point2
        """
    return np.linalg.norm(np.cross((point-line_point2),(point-line_point1)))/np.linalg.norm(line_point1 - line_point2)


def getPrincipalVectors(A): #
    """
        Get principal vectors and values of a matrix centered around (0,0,0)

        Args:
            A: np.narray (n,m)
                Matrix to extract principal vectors from

        Returns:
            Vectors: np.narray (m,m)
                The principal vectors from A
            Values: np.narray (m,m)
                The principal values from A
        """
    VT=np.linalg.eig(np.matmul(A.T,A))
    sort = sorted(zip(VT[0],VT[1].T.tolist()),reverse=True)
    values,vectors = zip(*sort)
    return vectors,values


def open3dpaint(nppoints, color_map = 'jet', reduce_for_vis = False, voxel_size = 0.1, pointsize = 0.1):
    """
        Opens an open3d visualizer and displays point clouds

        Args:
            nppoints: pclpy.pcl.PointCloud.PointXYZRGB | pclpy.pcl.PointCloud.PointXYZ | np.ndarray | list | tuple
                Either a (n,3) point cloud or a list or tuple of point clouds to be displayed
            
            color_map: str | list 3
                By default uses jet color map, it can be a list with 3 ints between 0 and 255 to represent an RBG color to color all points

            reduce_for_vis: bool
                If true it performs voxel subsampling before displaying the point cloud

            voxel_size: float
                If reduce_for_vis is true, sets the voxel size for the voxel subsampling

            pointsize: int
                Size of the distplayed points

        Returns:
            None
        """
    assert (type(nppoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(nppoints) == pclpy.pcl.PointCloud.PointXYZ) or (type(nppoints) == np.ndarray) or (type(nppoints) is list) or (type(nppoints) is tuple), 'Not valid point_cloud'
    
    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]
    try:
        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window()
        options = visualizer.get_render_option()
        options.background_color = np.asarray([0, 0, 0])
        options.point_size = pointsize

        if len(nppoints) > 1:
            for n,i in enumerate(nppoints):
                workpoints = i
                if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
                    workpoints = workpoints.xyz

                if reduce_for_vis:
                    workpoints = seg_tree.voxelize(workpoints,voxel_size)

                points = convertcloud(workpoints)
                color_coef = n/len(nppoints)/2 + n%2*.5
                if type(color_map) == np.ndarray:
                    color = color_map
                elif color_map == 'jet':
                    color=cm.jet(color_coef)[:3]
                else:
                    color=cm.Set1(color_coef)[:3]
                points.colors = open3d.utility.Vector3dVector(np.ones_like(workpoints)*color)
                #points.colors = open3d.utility.Vector3dVector(color)
                visualizer.add_geometry(points)
        else:
            workpoints = nppoints[0]
            if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
                workpoints = workpoints.xyz
                
            if reduce_for_vis:
                workpoints = seg_tree.voxelize(workpoints,voxel_size)
            points = convertcloud(workpoints)
            visualizer.add_geometry(points)
        visualizer.run()
        visualizer.destroy_window()
        
    except Exception as e:
        print(type(e))
        print(e.args)
        print(e)
        visualizer.destroy_window()
        
def plt3dpaint(nppoints, color_map = 'jet', reduce_for_vis = True, voxel_size = 0.2, pointsize = 0.1, subplots = 5):
    """
        displays point clouds on matplotlib 3d scatter plots

        Args:
            nppoints: pclpy.pcl.PointCloud.PointXYZRGB | pclpy.pcl.PointCloud.PointXYZ | np.ndarray | list | tuple
                Either a (n,3) point cloud or a list or tuple of point clouds to be displayed
            
            color_map: str | list 3
                By default uses jet color map, it can be a list with 3 ints between 0 and 255 to represent an RBG color to color all points

            reduce_for_vis: bool
                If true it performs voxel subsampling before displaying the point cloud

            voxel_size: float
                If reduce_for_vis is true, sets the voxel size for the voxel subsampling

            pointsize: int
                Size of the distplayed points

            subplots: int
                Number of subplots to create, each plot has a view rotation of 360/subplots

        Returns:
            None
        """
    assert (type(nppoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(nppoints) == pclpy.pcl.PointCloud.PointXYZ) or (type(nppoints) == np.ndarray) or (type(nppoints) is list) or (type(nppoints) is tuple), 'Not valid point_cloud'
    cloudlist = []
    cloudcolors = []
    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]
        
    if len(nppoints) > 1:
        for n,i in enumerate(nppoints):
            workpoints = i
            if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
                workpoints = workpoints.xyz

            if reduce_for_vis:
                workpoints = seg_tree.voxelize(workpoints,voxel_size)

            
            cloudmin = np.min(workpoints[:,2])
            cloudmax = np.max(workpoints[:,2])
    
            points = workpoints
            color_coef = n/len(nppoints)/2 + n%2*.5
            if type(color_map) == np.ndarray:
                color = color_map
            elif color_map == 'jet':
                color=cm.jet(color_coef)[:3]
            else:
                color=cm.Set1(color_coef)[:3]
            cloudcolors.append(np.ones_like(workpoints)*color + 0.4*(np.ones_like(workpoints) * ((workpoints[:,2] - cloudmin)/(cloudmax - cloudmin)).reshape(-1,1)-0.5) )
            cloudlist.append(points)
    else:
        workpoints = nppoints[0]
        if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
            workpoints = workpoints.xyz

        if reduce_for_vis:
            workpoints = seg_tree.voxelize(workpoints,voxel_size)
        cloudcolors.append(workpoints[:,2])
        cloudlist.append(workpoints)

    plt_pointcloud = np.concatenate(cloudlist)
    plt_colors = np.concatenate(cloudcolors)
    if len(nppoints) > 1:
        plt_colors = np.minimum(plt_colors,np.ones_like(plt_colors))
        plt_colors = np.maximum(plt_colors,np.zeros_like(plt_colors))
    fig = plt.figure(figsize=(30,16) )
    for i in range(subplots):
        ax = fig.add_subplot(1, subplots, i+1, projection='3d')
        ax.view_init(30, 360*i/subplots)
        ax.scatter3D(plt_pointcloud[:,0], plt_pointcloud[:,1], plt_pointcloud[:,2], c=plt_colors, s=pointsize)

        
        
def convertcloud(points):
    """
        Turns a numpy (n,3) point cloud to a open3d pointcloud

        Args:
            points: np.narray (n,3)
                A 3d numpy point cloud

        Returns:
            pcd: open3d.geometry.PointCloud
                An open 3d point cloud
        """
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    return pcd

def similarize(test, target):
    """
        Test a vectors angle to another vector and mirror its direction if it is greater than pi/2

        Args:
            test: np.narray (3)
                3d vector to test

            target: np.narray (3)
                3d vector to which test has to have an angle smaller than pi/2

        Returns:
            test: np.narray (3)
                3d vectors whos angle is below pi/2 with respect to the target vector
        """
    test = np.array(test)
    assert len(test) == 3,'vector must be dim 3'
    angle = angle_between_vectors(test,target)
    if angle > np.pi/2:
        test = -test
    return test