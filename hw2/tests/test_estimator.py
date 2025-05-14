import unittest
import numpy as np
import sys, os
# 將 src 加入模組搜尋路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from estimator import AffineEstimator

class FakeKeyPoint:
    def __init__(self, x, y):
        self.pt = (x, y)

class FakeMatch:
    def __init__(self, queryIdx, trainIdx):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx

class TestAffineEstimator(unittest.TestCase):
    def setUp(self):
        self.src_pts = np.array([[10,20],[30,60],[50,80],[90,40],[70,100]], dtype=np.float32)
        theta = np.deg2rad(15); s = 1.2
        c, s_a = np.cos(theta), np.sin(theta)
        self.A_true = np.array([[s*c, -s*s_a, 5],[s*s_a, s*c, -3]], dtype=np.float32)
        ones = np.ones((len(self.src_pts),1), dtype=np.float32)
        src_h = np.hstack([self.src_pts, ones])
        dst_pts = (self.A_true @ src_h.T).T
        self.kp1 = [FakeKeyPoint(x,y) for x,y in self.src_pts]
        self.kp2 = [FakeKeyPoint(x,y) for x,y in dst_pts]
        self.matches = [FakeMatch(i,i) for i in range(len(self.src_pts))]
        self.estimator = AffineEstimator(ransac_thresh=1.0, max_iters=1000)

    def test_affine_estimation(self):
        A_est, mask = self.estimator.estimate(self.kp1, self.kp2, self.matches)
        self.assertEqual(A_est.shape, (2,3))
        self.assertTrue(mask.sum() == len(self.matches))
        np.testing.assert_allclose(A_est, self.A_true, rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
    unittest.main()
