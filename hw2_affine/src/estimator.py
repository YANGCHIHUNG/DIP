"""
Module: estimator.py
功能：仿射矩陣估計（含 RANSAC）
"""
from typing import Tuple
import numpy as np
import cv2


def estimate_affine_partial(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_thresh: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 cv2.estimateAffinePartial2D 估計 2x3 仿射矩陣
    (保持比例、旋轉、平移，無透視)

    Args:
        src_pts: 來源點 (Nx2)
        dst_pts: 目標點 (Nx2)
        ransac_thresh: RANSAC 重投影誤差閾值

    Returns:
        M: 仿射矩陣 2x3
        mask: 內點掩碼 (Nx1)
    """
    # 注意：必須用 None 填 inliers，再依序提供 method 與 threshold
    M, mask = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        None,               # inliers output placeholder
        cv2.RANSAC,        # method
        ransac_thresh      # ransacReprojThreshold
    )
    return M, mask


def estimate_affine_full(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_thresh: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 cv2.estimateAffine2D 估計完整 2x3 仿射矩陣 (可包含剪切)

    Args:
        src_pts: 來源點 (Nx2)
        dst_pts: 目標點 (Nx2)
        ransac_thresh: RANSAC 重投影誤差閾值

    Returns:
        M: 仿射矩陣 2x3
        mask: 內點掩碼 (Nx1)
    """
    M, mask = cv2.estimateAffine2D(
        src_pts,
        dst_pts,
        None,               # inliers placeholder
        cv2.RANSAC,        # method
        ransac_thresh      # ransacReprojThreshold
    )
    return M, mask


def get_inlier_points(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根據掩碼分割內點

    Args:
        src_pts: 原始來源點 (Nx2)
        dst_pts: 原始目標點 (Nx2)
        mask: 由 RANSAC 產生的掩碼 (Nx1)

    Returns:
        in_src: 內點來源座標
        in_dst: 內點目標座標
    """
    mask_flat = mask.flatten()
    in_src = src_pts[mask_flat == 1]
    in_dst = dst_pts[mask_flat == 1]
    return in_src, in_dst


def estimate_affine(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    method: str = 'partial',
    ransac_thresh: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    高階介面：根據 method 選擇部分或完整仿射估計

    Args:
        src_pts: 來源點 (Nx2)
        dst_pts: 目標點 (Nx2)
        method: 'partial' 或 'full'
        ransac_thresh: RANSAC 重投影誤差閾值

    Returns:
        M: 仿射矩陣 2x3
        mask: 內點掩碼
    """
    method_lower = method.lower()
    if method_lower == 'partial':
        return estimate_affine_partial(src_pts, dst_pts, ransac_thresh)
    elif method_lower == 'full':
        return estimate_affine_full(src_pts, dst_pts, ransac_thresh)
    else:
        raise ValueError(f"Unknown estimation method: {method}")
