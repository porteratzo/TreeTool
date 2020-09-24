import pclpy
import numpy as np
from matplotlib import cm


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