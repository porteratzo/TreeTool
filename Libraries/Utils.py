import numpy as np
import pclpy
from matplotlib import cm
import matplotlib.pyplot as plt
import open3d
import Libraries.segTree as segTree

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

def angle_b_vectors(a,b):
    value = np.sum(np.multiply(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    if (value<-1) | (value>1):
        value = np.sign(value)
    angle = np.arccos(value)
    return angle

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

def DistPoint2Line(point,linepoint1, linepoint2=np.array([0,0,0])): #get minimum destance from a point to a line
    return np.linalg.norm(np.cross((point-linepoint2),(point-linepoint1)))/np.linalg.norm(linepoint1 - linepoint2)


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


def getPrincipalVectors(A): #get pricipal vectors and values of a matrix centered around (0,0,0)
    VT=np.linalg.eig(np.matmul(A.T,A))
    sort = sorted(zip(VT[0],VT[1].T.tolist()),reverse=True)
    Values,Vectors = zip(*sort)
    return Vectors,Values


def open3dpaint(nppoints, color = 'jet', reduce_for_Vis = False, voxelsize = 0.1, pointsize = 0.1):
    assert (type(nppoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(nppoints) == pclpy.pcl.PointCloud.PointXYZ) or (type(nppoints) == np.ndarray) or (type(nppoints) is list) or (type(nppoints) is tuple), 'Not valid point_cloud'
    
    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]
    try:
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = pointsize

        if len(nppoints) > 1:
            for n,i in enumerate(nppoints):
                workpoints = i
                if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
                    workpoints = workpoints.xyz

                if reduce_for_Vis:
                    workpoints = segTree.voxelize(workpoints,voxelsize)

                points = convertcloud(workpoints)
                colNORM = n/len(nppoints)/2 + n%2*.5
                if type(color) == np.ndarray:
                    pass
                elif color == 'jet':
                    color=cm.jet(colNORM)[:3]
                else:
                    color=cm.Set1(colNORM)[:3]
                points.colors = open3d.utility.Vector3dVector(np.ones_like(workpoints)*color)
                #points.colors = open3d.utility.Vector3dVector(color)
                vis.add_geometry(points)
        else:
            workpoints = nppoints[0]
            if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
                workpoints = workpoints.xyz
                
            if reduce_for_Vis:
                workpoints = segTree.voxelize(workpoints,voxelsize)
            points = convertcloud(workpoints)
            vis.add_geometry(points)
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(type(e))
        print(e.args)
        print(e)
        vis.destroy_window()
        
def plt3dpaint(nppoints, color = 'jet', reduce_for_Vis = True, voxelsize = 0.2, pointsize = 0.1, subplots = 5):
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

            if reduce_for_Vis:
                workpoints = segTree.voxelize(workpoints,voxelsize)

            
            cloudmin = np.min(workpoints[:,2])
            cloudmax = np.max(workpoints[:,2])
    
            points = workpoints
            colNORM = n/len(nppoints)/2 + n%2*.5
            if type(color) == np.ndarray:
                pass
            elif color == 'jet':
                color=cm.jet(colNORM)[:3]
            else:
                color=cm.Set1(colNORM)[:3]
            cloudcolors.append(np.ones_like(workpoints)*color + 0.4*(np.ones_like(workpoints) * ((workpoints[:,2] - cloudmin)/(cloudmax - cloudmin)).reshape(-1,1)-0.5) )
            cloudlist.append(points)
    else:
        workpoints = nppoints[0]
        if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
            workpoints = workpoints.xyz

        if reduce_for_Vis:
            workpoints = segTree.voxelize(workpoints,voxelsize)
        cloudcolors.append(workpoints[:,2])
        cloudlist.append(workpoints)

    PLTPC = np.concatenate(cloudlist)
    PLTCL = np.concatenate(cloudcolors)
    if len(nppoints) > 1:
        PLTCL = np.minimum(PLTCL,np.ones_like(PLTCL))
        PLTCL = np.maximum(PLTCL,np.zeros_like(PLTCL))
    fig = plt.figure(figsize=(30,16) )
    for i in range(subplots):
        ax = fig.add_subplot(1, subplots, i+1, projection='3d')
        ax.view_init(30, 360*i/subplots)
        ax.scatter3D(PLTPC[:,0], PLTPC[:,1], PLTPC[:,2], c=PLTCL, s=pointsize)

        
        
def convertcloud(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    #open3d.write_point_cloud(Path+'sync.ply', pcd)
    #pcd_load = open3d.read_point_cloud(Path+'sync.ply')
    return pcd

def similarize(vector,target):
    Nvector = np.array(vector)
    assert len(Nvector) == 3,'vector must be dim 3'
    angle = angle_b_vectors(Nvector,target)
    if angle > np.pi/2:
        Nvector = -Nvector
    return Nvector