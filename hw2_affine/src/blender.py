"""
Module: blender.py
功能：多頻段金字塔融合 (Laplacian Pyramid Blending)
"""
from typing import List, Tuple
import cv2
import numpy as np


def build_gaussian_pyramid(
    image: np.ndarray,
    levels: int
) -> List[np.ndarray]:
    """
    建立高斯金字塔。
    Args:
        image: 原始影像
        levels: 金字塔層數
    Returns:
        gauss_pyr: 由大到小的高斯金字塔列表
    """
    gauss_pyr = [image.copy()]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        gauss_pyr.append(image)
    return gauss_pyr


def build_laplacian_pyramid(
    gauss_pyr: List[np.ndarray]
) -> List[np.ndarray]:
    """
    建立拉普拉斯金字塔。
    Args:
        gauss_pyr: 高斯金字塔列表
    Returns:
        lap_pyr: 由大到小的拉普拉斯金字塔列表
    """
    lap_pyr: List[np.ndarray] = []
    for i in range(len(gauss_pyr) - 1):
        size = (gauss_pyr[i].shape[1], gauss_pyr[i].shape[0])
        expanded = cv2.pyrUp(gauss_pyr[i+1], dstsize=size)
        lap = cv2.subtract(gauss_pyr[i], expanded)
        lap_pyr.append(lap)
    lap_pyr.append(gauss_pyr[-1])  # 最底層直接加入
    return lap_pyr


def reconstruct_from_laplacian_pyramid(
    lap_pyr: List[np.ndarray]
) -> np.ndarray:
    """
    從拉普拉斯金字塔重建影像。
    Args:
        lap_pyr: 拉普拉斯金字塔列表
    Returns:
        重建後影像
    """
    image = lap_pyr[-1]
    for lev in reversed(lap_pyr[:-1]):
        size = (lev.shape[1], lev.shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(lev, image)
    return image


def multi_band_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray,
    levels: int = 4
) -> np.ndarray:
    """
    使用多頻段金字塔方法融合兩張影像。
    Args:
        img1: 影像 1
        img2: 影像 2
        mask: 融合遮罩 (灰階或浮點, 範圍 [0,1])
        levels: 金字塔層數
    Returns:
        blended: 融合後影像
    """
    # 建立高斯金字塔
    gp1 = build_gaussian_pyramid(img1, levels)
    gp2 = build_gaussian_pyramid(img2, levels)
    gp_mask = build_gaussian_pyramid(mask, levels)

    # 建立拉普拉斯金字塔
    lp1 = build_laplacian_pyramid(gp1)
    lp2 = build_laplacian_pyramid(gp2)

    # 融合金字塔
    lp_blend: List[np.ndarray] = []
    for l1, l2, gm in zip(lp1, lp2, gp_mask):
        # 若 mask 為單通道，expand to 3 channels
        if len(gm.shape) == 2:
            gm = cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR)
        blended = l1 * gm + l2 * (1.0 - gm)
        lp_blend.append(np.uint8(blended))

    # 重建融合影像
    blended = reconstruct_from_laplacian_pyramid(lp_blend)
    return blended
