import numpy as np
import pclpy
from pclpy import pcl

class IO:
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
        
    def pcl_to_numpy(cloud) -> np.array:
        # Convert to NumPy array
        points = np.zeros((cloud.size(), 3), dtype=np.float32)
        for i in range(cloud.size()):
            points[i, 0] = cloud.at(i).x  # X coordinate
            points[i, 1] = cloud.at(i).y  # Y coordinate
            points[i, 2] = cloud.at(i).z  # Z coordinate

        return points