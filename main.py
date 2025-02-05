import pcl
import numpy as np
from IO import IO

class PC_denoiser:
    @staticmethod
    def denoise_bilateral_filter(input_object, distance_sigma=0.1, normal_radius=0.2, spatial_sigma=0.1, output_file = None):
        cloud = IO.get_pcl_from(input=input_object)
        filter = cloud.make_bilateral_filter()

        # Set the parameters for the filter
        filter.set_DistanceSigma(distance_sigma) 
        filter.set_NormalRadius(normal_radius)  
        filter.set_SpatialSigma(spatial_sigma) 

        # Apply the filter
        filtered_cloud = filter.filter()

        if output_file:
            pcl.save(filtered_cloud, output_file)
        
        return filtered_cloud
    
    @staticmethod
    def denoise_voxel_grid_filter(input_object, leaf_size=0.1, output_file = None):
        cloud = IO.get_pcl_from(input=input_object)
        filter = cloud.make_voxel_grid_filter()

        # Set the parameters for the filter
        filter.set_leaf_size(leaf_size, leaf_size, leaf_size)

        # Apply the filter
        filtered_cloud = filter.filter()

        if output_file:
            pcl.save(filtered_cloud, output_file)
        
        return filtered_cloud
        
# Example usage
# arr = np.load("data/Tetrahedron.npy")
input_file = "data/Tetrahedron.npy"
output_file = 'denoised_point_cloud.pcd'

arr = np.load(input_file)
# print(type(arr))
arr = PC_denoiser.denoise_bilateral_filter(arr, output_file=output_file)
arr = IO.pcl_to_numpy(cloud=arr)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()