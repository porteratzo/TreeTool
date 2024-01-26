# %%
import open3d as o3d
#from porteratzolibs.visualization_o3d import open3dpaint

# %%


file_directory = r'data/downsampledlesscloudEURO1.pcd'
PointCloud = o3d.io.read_point_cloud(file_directory)

o3d.visualization.draw_geometries([PointCloud],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

app = o3d.visualization.gui.Application.instance
app.initialize()            
vis = o3d.visualization.O3DVisualizer("Open3DVisualizer")