import cv2
import numpy as np
from typing import Tuple, List


def detect_and_compute(img: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    偵測 SIFT 特徵點並計算描述子

    參數：
        img (np.ndarray): BGR 影像陣列

    回傳：
        kps (List[cv2.KeyPoint]): 偵測到的關鍵點列表
        desc (np.ndarray): 對應的特徵描述子陣列
    """
    # 轉為灰階影像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 建立 SIFT 偵測器
    sift = cv2.SIFT_create()
    # 偵測關鍵點與計算描述子
    kps, desc = sift.detectAndCompute(gray, None)
    # 強制轉為 list，避免 OpenCV 回傳 tuple
    kps = list(kps)
    return kps, desc