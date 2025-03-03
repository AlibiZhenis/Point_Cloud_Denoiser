import unittest
import numpy as np
from PC_denoiser import PC_denoiser

from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

input_file = "data/Tetrahedron.npy"
output_file = "data/temp.npy"

class Test_PC_Denoiser(unittest.TestCase):
    def setUp(self) -> None:
        self.cloud = np.load(input_file)
        return super().setUp()
    
    def tearDown(self) -> None:
        del self.cloud
        return super().tearDown()
    
    def test_mls_with_arr(self):
        filtered_cloud = PC_denoiser.denoise_mls(self.cloud, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)
    
    def test_mls_with_file(self):
        filtered_cloud = PC_denoiser.denoise_mls(input_file, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)

    def test_voxelgrid_with_arr(self):
        filtered_cloud = PC_denoiser.denoise_voxel_grid(self.cloud, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)
    
    def test_voxelgrid_with_file(self):
        filtered_cloud = PC_denoiser.denoise_voxel_grid(input_file, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)

    def test_pointfilter_with_arr(self):
        filtered_cloud = PC_denoiser.denoise_pointfilter(self.cloud, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)
    
    def test_pointfilter_with_file(self):
        filtered_cloud = PC_denoiser.denoise_pointfilter(input_file, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)

    def test_scorebased_with_arr(self):
        filtered_cloud = PC_denoiser.denoise_score_based(self.cloud, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)
    
    def test_scorebased_with_file(self):
        filtered_cloud = PC_denoiser.denoise_score_based(input_file, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)

    def test_pcpnet_with_arr(self):
        filtered_cloud = PC_denoiser.denoise_pointcleannet(self.cloud, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)
    
    def test_pcpnet_with_file(self):
        filtered_cloud = PC_denoiser.denoise_pointcleannet(input_file, output_file=output_file)
        self.assertTrue(filtered_cloud.shape[0] != 0)
        self.assertTrue(filtered_cloud.shape[1] != 0)
