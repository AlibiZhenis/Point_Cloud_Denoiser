import os
# import pcl
import numpy as np
import torch

# from IO import IO
from PointFilter.Pointfilter_Network_Architecture import pointfilternet
from PointFilter.Pointfilter_DataLoader import PointcloudPatchDataset

file_dir = os.path.dirname(os.path.abspath(__file__)) 

class PC_denoiser:
    # @staticmethod
    # def denoise_bilateral_filter(input_object, distance_sigma=0.1, normal_radius=0.2, spatial_sigma=0.1, output_file = None):
    #     cloud = IO.get_pcl_from(input=input_object)
    #     filter = cloud.make_bilateral_filter()

    #     # Set the parameters for the filter
    #     filter.set_DistanceSigma(distance_sigma) 
    #     filter.set_NormalRadius(normal_radius)  
    #     filter.set_SpatialSigma(spatial_sigma) 

    #     # Apply the filter
    #     filtered_cloud = filter.filter()

    #     if output_file:
    #         pcl.save(filtered_cloud, output_file)
        
    #     return filtered_cloud
    
    # @staticmethod
    # def denoise_voxel_grid_filter(input_object, leaf_size=0.1, output_file = None):
    #     cloud = IO.get_pcl_from(input=input_object)
    #     filter = cloud.make_voxel_grid_filter()

    #     # Set the parameters for the filter
    #     filter.set_leaf_size(leaf_size, leaf_size, leaf_size)

    #     # Apply the filter
    #     filtered_cloud = filter.filter()

    #     if output_file:
    #         pcl.save(filtered_cloud, output_file)
        
    #     return filtered_cloud
    
    @staticmethod
    def denoise_pointfilter(cloud, patch_radius = 0.05, num_workers = 0, model_path = None, output_file = None):
        # cloud = IO.get_pcl_from(input=input_object)

        test_dataset = PointcloudPatchDataset(
            cloud=cloud,
            patch_radius=patch_radius,
            train_state='evaluation',
            shape_name="dummy")
        # print(len(test_dataset))
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=int(num_workers),
            batch_size = 64,
            )
        # print(len(test_dataloader))
        
        pointfilter_eval = pointfilternet().cuda()
        if model_path is None:
            model_path = os.path.join(file_dir, 'PointFilter\Summary\pre_train_model\model_full_ae.pth')
        checkpoint = torch.load(model_path)
        pointfilter_eval.load_state_dict(checkpoint['state_dict'])
        pointfilter_eval.cuda()
        pointfilter_eval.eval()

        patch_radius = test_dataset.patch_radius_absolute
        pred_pts = np.empty((0, 3), dtype='float32')
        for batch_ind, data_tuple in enumerate(test_dataloader):
            noise_patch, noise_inv, noise_disp = data_tuple
            # print(noise_patch.shape, noise_inv.shape, noise_disp.shape)
            noise_patch = noise_patch.float().cuda()
            noise_inv = noise_inv.float().cuda()
            noise_patch = noise_patch.transpose(2, 1).contiguous()
            # print(noise_patch.shape, noise_inv.shape, noise_disp.shape)
            predict = pointfilter_eval(noise_patch)
            predict = predict.unsqueeze(2)
            predict = torch.bmm(noise_inv, predict)
            pred_pts = np.append(pred_pts,
                                    np.squeeze(predict.data.cpu().numpy()) * patch_radius + noise_disp.numpy(),
                                    axis=0)
        # np.save(os.path.join(opt.save_dir, shape_name + '_pred_iter_' + str(eval_index + 1) + '.npy'),
        #     pred_pts.astype('float32'))


        if output_file:
            # pcl.save(pred_pts.astype('float32'), output_file)
            np.save(output_file, pred_pts.astype('float32'))
        
        return pred_pts
        
# Example usage
# arr = np.load("data/Tetrahedron.npy")
input_file = "PointFilter/Dataset/Test/boxunion2_100K_0.005.npy"
output_file = 'denoised_point_cloud.pcd'

arr = np.load(input_file)
print(arr.shape)
arr = PC_denoiser.denoise_pointfilter(arr, output_file=output_file)
print(arr.shape)
# arr = IO.pcl_to_numpy(cloud=arr)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2])

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

import open3d as o3d
import numpy as np

# Create a sample point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(arr)

# Optionally, you can set colors for the points
colors = np.random.rand(100, 3)  # Random colors for each point
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])