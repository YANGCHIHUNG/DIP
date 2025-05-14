import cv2
import numpy as np
from estimator import AffineEstimator
from feature import detect_and_compute
from matcher import match_features
from blender import Blender

class Stitcher:
    """
    Stitcher 使用 Affine Transform 進行全景拼接。
    """
    def __init__(self, ransac_thresh: float = 3.0, max_iters: int = 2000):
        self.estimator = AffineEstimator(ransac_thresh, max_iters)
        self.blender = Blender()

    def stitch(self, images: list[np.ndarray]) -> np.ndarray:
        """
        將多張影像串接為全景圖。

        Args:
            images: BGR 影像陣列列表，依拍攝順序排列

        Returns:
            全景圖像
        """
        # 1. 特徵偵測與描述子計算
        feats = [detect_and_compute(img) for img in images]
        kps_list, descs_list = zip(*feats)

        # 2. 特徵配對
        matches_list = [
            match_features(descs_list[i], descs_list[i+1])
            for i in range(len(images) - 1)
        ]

        # 3. 仿射矩陣估算
        transforms = []
        for i, matches in enumerate(matches_list):
            A, mask = self.estimator.estimate(
                kps_list[i], kps_list[i+1], matches
            )
            transforms.append(A)

        # 4. 計算畫布邊界與平移
        h0, w0 = images[0].shape[:2]
        corners = np.float32([[0,0],[w0,0],[w0,h0],[0,h0]]).reshape(-1,1,2)
        acc_H = np.eye(3)
        all_corners = [cv2.perspectiveTransform(corners, acc_H)]
        for A in transforms:
            H = np.vstack([A, [0,0,1]])
            acc_H = acc_H @ H
            all_corners.append(cv2.perspectiveTransform(corners, acc_H))
        all_pts = np.concatenate(all_corners, axis=0)
        [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        translation = [-xmin, -ymin]
        pano_size = (xmax - xmin, ymax - ymin)

        # 5. 首張影像放置
        M0 = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]]
        ], dtype=np.float32)
        panorama = cv2.warpAffine(images[0], M0, pano_size)

        # 6. 依序對後續影像執行仿射投影並融合
        acc_H = np.eye(3)
        for idx, img in enumerate(images[1:]):
            A = transforms[idx]
            H = np.vstack([A, [0,0,1]])
            acc_H = acc_H @ H
            H_off = acc_H.copy()
            H_off[0,2] += translation[0]
            H_off[1,2] += translation[1]
            warp_A = H_off[:2]
            warped = cv2.warpAffine(img, warp_A, pano_size)
            panorama = self.blender.blend(panorama, warped)

        return panorama


def stitch_images(
    images: list[np.ndarray],
    ransac_thresh: float = 3.0,
    max_iters: int = 2000
) -> np.ndarray:
    """
    以預載影像列表執行拼接，並回傳全景圖。

    Args:
        images: 已讀取的 BGR 影像列表
        ransac_thresh: RANSAC 重投影閾值
        max_iters: RANSAC 最大迭代次數

    Returns:
        全景拼接後的影像陣列
    """
    stitcher = Stitcher(ransac_thresh, max_iters)
    return stitcher.stitch(images)
