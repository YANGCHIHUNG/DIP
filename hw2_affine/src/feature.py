"""
Module: feature.py
功能：特徵偵測與描述（Keypoints & Descriptors）
"""
from typing import Tuple, List, Optional
import cv2
import numpy as np


def create_feature_detector(
    method: str = 'ORB',
    **kwargs
) -> cv2.Feature2D:
    """
    初始化特徵偵測器。

    Args:
        method: 偵測方法，可選 'SIFT', 'ORB', 'SURF'
        **kwargs: 傳給對應 create 函式的參數

    Returns:
        OpenCV Feature2D 物件
    """
    method_upper = method.upper()
    if method_upper == 'SIFT':
        # SIFT 無法在某些 OpenCV 版本中預設啟用，請確認 contrib 模組已安裝
        detector = cv2.SIFT_create(**kwargs)
    elif method_upper == 'SURF':
        # 需安裝 xfeatures2d
        detector = cv2.xfeatures2d.SURF_create(**kwargs)
    elif method_upper == 'ORB':
        detector = cv2.ORB_create(**kwargs)
    else:
        raise ValueError(f"Unknown feature detection method: {method}")
    return detector


def detect_and_compute(
    image: np.ndarray,
    detector: cv2.Feature2D
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    對影像進行特徵偵測與描述。

    Args:
        image: 輸入影像，可為灰階或彩色
        detector: 已初始化的 Feature2D 偵測器

    Returns:
        keypoints: 特徵點列表
        descriptors: 特徵描述子 (NumPy 陣列)
    """
    # 如果是彩色圖，需先轉為灰階
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image

    keypoints, descriptors = detector.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def extract_features(
    image: np.ndarray,
    method: str = 'ORB',
    detector_params: Optional[dict] = None
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    一次完成特徵偵測與描述的高階函式。

    Args:
        image: 輸入影像
        method: 偵測方法 ('SIFT', 'SURF', 'ORB')
        detector_params: 傳遞給 create_feature_detector 的參數

    Returns:
        keypoints: 特徵點列表
        descriptors: 特徵描述子 (NumPy 陣列)
    """
    params = detector_params or {}
    detector = create_feature_detector(method, **params)
    return detect_and_compute(image, detector)
