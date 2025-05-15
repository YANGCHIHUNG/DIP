import cv2
import numpy as np
from typing import List


def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75) -> List[cv2.DMatch]:
    """
    使用 BFMatcher 與 Lowe's ratio test 進行特徵匹配

    參數：
        desc1 (np.ndarray): 第一張影像的描述子陣列
        desc2 (np.ndarray): 第二張影像的描述子陣列
        ratio (float): Lowe's ratio test 閾值

    回傳：
        matches (List[cv2.DMatch]): 過濾後的有效匹配對列表
    """
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good