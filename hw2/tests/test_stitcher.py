import unittest
import numpy as np
import cv2
import sys, os
# 將 src 加入模組搜尋路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from stitcher import Stitcher

class TestStitcherAffine(unittest.TestCase):
    def setUp(self):
        self.img1 = np.zeros((100,100,3), dtype=np.uint8)
        cv2.rectangle(self.img1, (30,30), (70,70), (255,255,255), -1)
        M = np.array([[1,0,20],[0,1,10]], dtype=np.float32)
        self.img2 = cv2.warpAffine(self.img1, M, (100,100))
        self.stitcher = Stitcher(ransac_thresh=5.0, max_iters=500)

    def test_output_shape_and_type(self):
        pano = self.stitcher.stitch([self.img1, self.img2])
        expected_h = 100 + 10
        expected_w = 100 + 20
        self.assertEqual(pano.shape, (expected_h, expected_w, 3))
        self.assertEqual(pano.dtype, np.uint8)

    def test_panorama_contains_white_pixels(self):
        pano = self.stitcher.stitch([self.img1, self.img2])
        self.assertTrue(np.all(pano[50,50] == [255,255,255]))
        self.assertTrue(np.all(pano[60,70] == [255,255,255]))

if __name__ == '__main__':
    unittest.main()
