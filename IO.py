import numpy as np
import pclpy
from pclpy import pcl
import open3d as o3d
from PIL import Image


class IO:
    @classmethod
    def get_pcl_from(cls, input):
        if type(input) == str:
            if input[-4:] == ".npy":
                arr = np.load(input)
                cloud = pcl.PointCloud.PointXYZ(arr.astype(np.float32))
            elif input[-4:] == ".ply":
                cloud = pcl.PointCloud.PointXYZ()
                pcl.io.loadPLYFile(input, cloud)
            elif input[-4:] == ".pcd":
                cloud = pcl.PointCloud.PointXYZ()
                pcl.io.loadPCDFile(input, cloud)
            else:
                raise NotImplementedError(input)
            return cloud
        elif type(input) == np.ndarray:
            cloud = pcl.PointCloud.PointXYZ(input.astype(np.float32))
            return cloud
        else:
            raise NotImplementedError(input)
        
    @classmethod
    def save_to_file(cls, input, filename):
        np.savetxt(filename, input)
    
    @classmethod
    def get_arr_from(cls, input) -> np.ndarray:
        if (type(input) == str) and input[-4:] == ".npy":
            return np.load(input)
        elif (type(input) == np.ndarray):
            return input
        return cls.pcl_to_numpy(cls.get_pcl_from(input))
    
    @classmethod
    def pcl_to_numpy(cls, cloud) -> np.array:
        # Convert to NumPy array
        points = np.zeros((cloud.size(), 3), dtype=np.float32)
        for i in range(cloud.size()):
            points[i, 0] = cloud.at(i).x  # X coordinate
            points[i, 1] = cloud.at(i).y  # Y coordinate
            points[i, 2] = cloud.at(i).z  # Z coordinate

        return points.astype(np.float32)
    
    @classmethod
    def visualize_arr(cls, cloud: np.ndarray) -> None:
        # Create a sample point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cloud)

        # Optionally, you can set colors for the points
        colors = np.random.rand(100, 3)  # Random colors for each point
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])

    @classmethod
    def save_visualization(cls, cloud: np.ndarray, save_path: str) -> None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cloud)

        # Optionally, you can set colors for the points
        colors = np.random.rand(100, 3)  # Random colors for each point
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Create a visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)

        # Add the point cloud to the visualizer
        vis.add_geometry(point_cloud)

        # Update the visualizer to render
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # Capture the image
        image = vis.capture_screen_float_buffer(do_render=True)
        # vis.capture_screen_image("point_cloud_visualization.png")

        # Convert the image to a format suitable for saving
        image = (np.asarray(image) * 255).astype(np.uint8)

        # # Save the image
        img = Image.fromarray(image)
        img.save(save_path)

        # Close the visualizer
        vis.destroy_window()

    @classmethod
    def visualize_pcl(cls, cloud) -> None:
        # Create a sample point cloud
        arr = IO.pcl_to_numpy(cloud=cloud)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(arr)

        # Optionally, you can set colors for the points
        colors = np.random.rand(100, 3)  # Random colors for each point
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud])