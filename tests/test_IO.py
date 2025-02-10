import unittest
import numpy as np
from IO import IO
import pclpy

input_file = "data/Tetrahedron.npy"
output_file = "data/temp.npy"

class Test_IO(unittest.TestCase):
    
    def test_get_pcl_from_npy(self):
        cloud = IO.get_pcl_from(input_file)
        self.assertIsInstance(cloud, pclpy.pcl.PointCloud.PointXYZ)
        self.assertTrue(cloud.size() != 0)

    def test_get_arr_from_npy(self):
        cloud = IO.get_arr_from(input_file)
        self.assertIsInstance(cloud, np.ndarray)
        self.assertTrue(cloud.shape[0] != 0)