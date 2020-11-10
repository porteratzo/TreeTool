import pclpy
import numpy as np
from matplotlib import cm
import open3d
import segTree


def PCL3dpaint(nppoints,axis=None):
    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]
    try:
        newv = pclpy.pcl.visualization.PCLVisualizer()
        if axis is not None:
            newv.addCoordinateSystem(axis);
        for n,i in enumerate(nppoints):
            if type(i) == pclpy.pcl.PointCloud.PointXYZRGB:
                newv.addPointCloud(i,'PC'+str(n))
            elif type(i) == pclpy.pcl.PointCloud.PointXYZ:
                colNORM = n / len(nppoints) / 2 + n % 2 * .5
                color = np.uint8( np.array( cm.jet( colNORM)[:3]) * 255)
                colorHandeler = pclpy.pcl.visualization.PointCloudColorHandlerCustom.PointXYZ(i,color[0],color[1],color[2])
                newv.addPointCloud(i,colorHandeler,'PC'+str(n))
            else:
                colNORM = n / len(nppoints) / 2 + n % 2 * .5
                color = np.uint8( np.array( cm.jet( colNORM)[:3]) * 255)
                pointcloud = pclpy.pcl.PointCloud.PointXYZ(i)
                colorHandeler = pclpy.pcl.visualization.PointCloudColorHandlerCustom.PointXYZ(pointcloud,color[0],color[1],color[2])
                newv.addPointCloud(pointcloud,colorHandeler,'PC'+str(n))

        while not newv.wasStopped():
            newv.spinOnce(10)
        newv.close()
        
    except Exception as e:
        newv.close()
        print(e)


def open3dpaint(nppoints, color = 'jet', reduce_for_Vis = False, voxelsize = 0.1):
    assert (type(nppoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(nppoints) == pclpy.pcl.PointCloud.PointXYZ) or (type(nppoints) == np.ndarray) or (type(nppoints) is list) or (type(nppoints) is tuple), 'Not valid pointcloud'
    
    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]
    try:
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])

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
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(e)
        vis.destroy_window()
        
        
def convertcloud(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    #open3d.write_point_cloud(Path+'sync.ply', pcd)
    #pcd_load = open3d.read_point_cloud(Path+'sync.ply')
    return pcd