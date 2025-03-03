import os
import numpy as np
import torch
import open3d as o3d
from tqdm.auto import tqdm

import pclpy
from pclpy import pcl 

from IO import IO
from PointFilter.Pointfilter_Network_Architecture import pointfilternet
from PointFilter.Pointfilter_DataLoader import PointcloudPatchDataset

from score_denoise.utils.misc import *
from score_denoise.utils.denoise import *
from score_denoise.models.denoise import *

from pointcleannet.noise_removal.eval_pcpnet import eval_pcpnet

file_dir = os.path.dirname(os.path.abspath(__file__)) 

class PC_denoiser:
    @classmethod
    def denoise_mls(cls, cloud, search_radius=0.05, compute_normals=True, num_threads=8, output_file=None):
        pcl_cloud = IO.get_pcl_from(cloud)
        filtered_cloud = pcl_cloud.moving_least_squares(search_radius=search_radius, compute_normals=compute_normals, num_threads=num_threads)

        filtered_points = IO.pcl_to_numpy(filtered_cloud)

        if output_file:
            np.save(output_file, filtered_points.astype('float32'))
        
        return filtered_points

    @classmethod
    def denoise_voxel_grid(cls, cloud, leaf_size = 0.1, output_file=None):
        pcl_cloud = IO.get_pcl_from(cloud)

        # Create the voxel grid filter
        voxel_grid_filter = pcl.filters.VoxelGrid.PointXYZ()

        # Set the leaf size (the size of the voxel)
        voxel_grid_filter.setLeafSize(leaf_size, leaf_size, leaf_size)

        # Apply the filter
        filtered_cloud = pcl.PointCloud.PointXYZ()
        voxel_grid_filter.setInputCloud(pcl_cloud)
        voxel_grid_filter.filter(filtered_cloud)

        # Convert the filtered point cloud back to a NumPy array if needed
        filtered_points = IO.pcl_to_numpy(filtered_cloud)

        if output_file:
            np.save(output_file, filtered_points.astype('float32'))
        
        return filtered_points

    @classmethod
    def denoise_pointfilter(cls, input, patch_radius = 0.05, num_workers = 0, model_path = None, output_file = None):
        cloud = IO.get_arr_from(input)

        test_dataset = PointcloudPatchDataset(
            cloud=cloud,
            patch_radius=patch_radius,
            train_state='evaluation',
            shape_name="dummy")
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            num_workers=int(num_workers),
            batch_size = 64,
            )
                
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
            noise_patch = noise_patch.float().cuda()
            noise_inv = noise_inv.float().cuda()
            noise_patch = noise_patch.transpose(2, 1).contiguous()
            predict = pointfilter_eval(noise_patch)
            predict = predict.unsqueeze(2)
            predict = torch.bmm(noise_inv, predict)
            pred_pts = np.append(pred_pts,
                                    np.squeeze(predict.data.cpu().numpy()) * patch_radius + noise_disp.numpy(),
                                    axis=0)


        if output_file:
            np.save(output_file, pred_pts.astype('float32'))
        
        return pred_pts
    
    @classmethod
    def denoise_score_based(cls, input, cluster_size=30000, model_path = None, output_file=None):
        if model_path is None:
            model_path = os.path.join(file_dir, "score_denoise\pretrained\ckpt.pt")
        device = "cuda"

        ckpt = torch.load(model_path, map_location=device)
        model = DenoiseNet(ckpt['args']).to(device)
        model.load_state_dict(ckpt['state_dict'])

        cloud = IO.get_arr_from(input)
        cloud = torch.FloatTensor(cloud)

        if cloud.size(0) <= 50000:
            pcl, center, scale = NormalizeUnitSphere.normalize(cloud)
            pcl = pcl.to(device)
            pcl_denoised = patch_based_denoise(model, pcl).cpu()
            pcl_denoised = pcl_denoised * scale + center
        else:
            pcl = cloud.to(device)
            pcl_denoised = denoise_large_pointcloud(
                model=model,
                pcl=pcl,
                cluster_size=cluster_size,
                seed=2025,
            )
            pcl_denoised = pcl_denoised.cpu().numpy()

        if output_file:
            np.save(output_file, pcl_denoised.astype('float32'))
        
        return pcl_denoised
    
    @classmethod
    def denoise_pointcleannet(cls, input, output_file = None, model_path = None, verbose=False):
        arr = IO.get_arr_from(input)
        filename = "data\\temp.npy"
        IO.save_to_file(arr, filename)

        output_cloud = eval_pcpnet(input_filename=filename, batchSize=64, verbose=verbose)
        if output_file:
            np.save(output_file, output_cloud.astype('float32'))

        return output_cloud