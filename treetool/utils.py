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
import open3d


def rotation_matrix_from_vectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
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
    if all(np.abs(vector1) == np.abs(vector2)):
        return np.eye(3)
    a, b = (vector1 / np.linalg.norm(vector1)).reshape(3), (
        vector2 / np.linalg.norm(vector2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + matrix + matrix.dot(matrix) * ((1 - c) / (s**2))
    return rotation_matrix


def angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
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
    value = np.sum(np.multiply(vector1, vector2)) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    if (value < -1) | (value > 1):
        value = np.sign(value)
    angle = np.arccos(value)
    return angle


def makecylinder(model: np.ndarray = [0, 0, 0, 1, 0, 0, 1], height: float = 1, density: int = 10) -> np.ndarray:
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
    X, Y, Z = model[:3]
    # get 3d points to make an upright cylinder centered to the origin
    n = np.arange(0, 360, int(360 / density))
    height = np.arange(0, height, height / density)
    n = np.deg2rad(n)
    x, z = np.meshgrid(n, height)
    x = x.flatten()
    z = z.flatten()
    cyl = np.vstack([np.cos(x) * radius, np.sin(x) * radius, z]).T
    # rotate and translate the cylinder to fit the model
    rotation = rotation_matrix_from_vectors([0, 0, 1], model[3:6])
    rotated_cylinder = np.matmul(rotation, cyl.T).T + np.array([X, Y, Z])
    return rotated_cylinder


def DistPoint2Line(point: np.ndarray, line_point1: np.ndarray, line_point2: np.ndarray = np.array([0, 0, 0])) -> float:
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
    return np.linalg.norm(np.cross((point - line_point2), (point - line_point1))) / np.linalg.norm(
        line_point1 - line_point2
    )


def getPrincipalVectors(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  #
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
    VT = np.linalg.eig(np.matmul(A.T, A))
    sort = sorted(zip(VT[0], VT[1].T.tolist()), reverse=True)
    values, vectors = zip(*sort)
    return vectors, values


def convertcloud(points: np.ndarray) -> open3d.geometry.PointCloud:
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


def makesphere(centroid: list[float] = [0, 0, 0], radius: float = 1, dense: int = 90) -> np.ndarray:
    n = np.arange(0, 360, int(360 / dense))
    n = np.deg2rad(n)
    x, y = np.meshgrid(n, n)
    x = x.flatten()
    y = y.flatten()
    sphere = np.vstack(
        [
            centroid[0] + np.sin(x) * np.cos(y) * radius,
            centroid[1] + np.sin(x) * np.sin(y) * radius,
            centroid[2] + np.cos(x) * radius,
        ]
    ).T
    return sphere


def similarize(test: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    assert len(test) == 3, "vector must be dim 3"
    angle = angle_between_vectors(test, target)
    if angle > np.pi / 2:
        test = -test
    return test


def Iscaled_dimensions(las_file, new_data: dict) -> np.ndarray:

    x_dimension = np.array(new_data["X"])
    scale = las_file.header.scales[0]
    offset = las_file.header.offsets[0]
    x = x_dimension + offset

    y_dimension = np.array(new_data["Y"])
    offset = las_file.header.offsets[1]
    y = y_dimension + offset

    z_dimension = np.array(new_data["Z"])
    offset = las_file.header.offsets[2]
    z = z_dimension + offset
    return np.vstack([x, y, z]).T


def scaled_dimensions(
    las_file,
) -> np.ndarray:
    xyz = las_file.xyz
    x_dimension = xyz[:, 0]
    offset = las_file.header.offsets[0]
    x = x_dimension - offset

    y_dimension = xyz[:, 1]
    offset = las_file.header.offsets[1]
    y = y_dimension - offset

    z_dimension = xyz[:, 2]
    offset = las_file.header.offsets[2]
    z = z_dimension - offset
    return np.vstack([x, y, z]).T
