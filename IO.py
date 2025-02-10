import numpy as np
import pclpy
from pclpy import pcl
import open3d as o3d


class IO:
    @staticmethod
    def get_pcl_from(input):
        if type(input) == str:
            if input[-4:] == ".npy":
                arr = np.load(input)
                cloud = pcl.PointCloud.PointXYZ()
                cloud.from_array(arr.astype(np.float32))
            elif input[-4:] in [".pcd", ".ply"]:
                cloud = pcl.load(input)
            else:
                raise NotImplementedError(input)
            return cloud
        elif type(input) == np.ndarray:
            cloud = pcl.PointCloud.PointXYZ()
            cloud.from_array(input.astype(np.float32))
            return cloud
        else:
            raise NotImplementedError(input)
    
    @staticmethod
    def pcl_to_numpy(cloud) -> np.array:
        # Convert to NumPy array
        points = np.zeros((cloud.size(), 3), dtype=np.float32)
        for i in range(cloud.size()):
            points[i, 0] = cloud.at(i).x  # X coordinate
            points[i, 1] = cloud.at(i).y  # Y coordinate
            points[i, 2] = cloud.at(i).z  # Z coordinate

        return points
    
    @staticmethod
    def visualize_arr(cloud: np.ndarray) -> None:
        # Create a sample point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cloud)

        # Optionally, you can set colors for the points
        colors = np.random.rand(100, 3)  # Random colors for each point
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])

    @staticmethod
    def visualize_pcl(cloud) -> None:
        # Create a sample point cloud
        arr = IO.pcl_to_numpy(cloud=cloud)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(arr)

        # Optionally, you can set colors for the points
        colors = np.random.rand(100, 3)  # Random colors for each point
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])