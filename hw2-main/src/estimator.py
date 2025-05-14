import cv2
import numpy as np
from typing import List


def find_homography(
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    ransac_thresh: float
) -> np.ndarray:
    """
    根據匹配點計算單應性矩陣 H，使用 RANSAC 排除外點

    參數：
        kps1 (List[cv2.KeyPoint]): 基準影像關鍵點列表
        kps2 (List[cv2.KeyPoint]): 待拼影像關鍵點列表
        matches (List[cv2.DMatch]): 對應的匹配對列表
        ransac_thresh (float): RANSAC 重投影誤差閾值

    回傳：
        H (np.ndarray): 3x3 單應性矩陣
    """
    # 擷取對應點坐標
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    # 計算 Homography
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)
    return H