import cv2
import numpy as np

class AffineEstimator:
    """
    AffineEstimator 使用 RANSAC 方法估算仿射變換矩陣。

    Attributes:
        ransac_thresh (float): RANSAC 重投影誤差閾值。
        max_iters (int): RANSAC 最大迭代次數。
    """
    def __init__(self, ransac_thresh: float = 3.0, max_iters: int = 2000):
        self.ransac_thresh = ransac_thresh
        self.max_iters = max_iters

    def estimate(self, keypoints1, keypoints2, matches):
        """
        依據特徵點對估算仿射矩陣。

        Args:
            keypoints1: 第一張影像的 keypoints 列表 (cv2.KeyPoint)
            keypoints2: 第二張影像的 keypoints 列表 (cv2.KeyPoint)
            matches: cv2.DMatch 物件列表，表示描述子配對結果

        Returns:
            A (np.ndarray): 2x3 仿射變換矩陣
            mask (np.ndarray): 內點遮罩，shape=(N,1)，值為 1 表示內點
        """
        if len(matches) < 3:
            # fallback: 當 match 少於 3 時，僅以平移轉換為解
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            # 平移向量由所有匹配點的位移中位數決定
            diffs = dst_pts - src_pts
            tx = float(np.median(diffs[:, 0]))
            ty = float(np.median(diffs[:, 1]))
            A = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
            mask = np.ones((len(matches), 1), dtype=np.uint8)
            return A, mask

        # 構造匹配點陣列
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # 使用 RANSAC 估算仿射矩陣
        A, mask = cv2.estimateAffine2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=self.max_iters
        )
        if A is None:
            raise RuntimeError("Affine transform estimation failed.")

        return A, mask
