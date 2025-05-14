"""
Module: matcher.py
功能：特徵匹配與篩選
"""
from typing import List, Tuple
import cv2
import numpy as np


def create_bf_matcher(
    norm: int = cv2.NORM_HAMMING,
    cross_check: bool = True
) -> cv2.DescriptorMatcher:
    """
    建立暴力匹配器 (Brute-Force Matcher)
    Args:
        norm: 距離度量 (預設 NORM_HAMMING)
        cross_check: 是否啟用交叉檢查
    Returns:
        cv2.DescriptorMatcher
    """
    return cv2.BFMatcher(norm, cross_check)


def create_flann_matcher(
    algorithm: int = 1,
    trees: int = 5
) -> cv2.DescriptorMatcher:
    """
    建立 FLANN 匹配器
    Args:
        algorithm: 索引算法 (1=KDTree, 0=LSH)
        trees: KDTree 樹數量
    Returns:
        cv2.FlannBasedMatcher
    """
    if algorithm == 1:
        index_params = dict(algorithm=1, trees=trees)
    else:
        index_params = dict(algorithm=0, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def match_descriptors(
    matcher: cv2.DescriptorMatcher,
    desc1: np.ndarray,
    desc2: np.ndarray
) -> List[cv2.DMatch]:
    """
    執行 descriptor 匹配
    """
    return matcher.match(desc1, desc2)


def knn_match_descriptors(
    matcher: cv2.DescriptorMatcher,
    desc1: np.ndarray,
    desc2: np.ndarray,
    k: int = 2
) -> List[List[cv2.DMatch]]:
    """
    執行 KNN 匹配
    """
    return matcher.knnMatch(desc1, desc2, k=k)


def filter_matches_ratio(
    knn_matches: List[List[cv2.DMatch]],
    ratio: float = 0.75
) -> List[cv2.DMatch]:
    """
    使用 Lowe's ratio test 過濾匹配
    Args:
        knn_matches: KNN 匹配結果
        ratio: 比率閾值
    Returns:
        符合條件的匹配列表
    """
    good = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def filter_matches_by_top_n(
    matches: List[cv2.DMatch],
    n: int = 50
) -> List[cv2.DMatch]:
    """
    保留最好的 N 個匹配
    """
    return sorted(matches, key=lambda x: x.distance)[:n]


def get_matched_points(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根據 matches 取得對應的點座標矩陣
    Returns:
        pts1, pts2: 形狀 (N, 2) 的座標陣列
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2
