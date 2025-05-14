"""
Module: image_utils.py
功能：影像輔助處理（遮罩、裁剪、羽化、填邊）
"""
from typing import Tuple
import numpy as np
import cv2


def create_mask_from_image(
    image: np.ndarray
) -> np.ndarray:
    """
    將影像中的非零像素區域轉為二元遮罩。

    Args:
        image: 輸入影像，彩色或灰階
    Returns:
        mask: 0/1 二元遮罩
    """
    if image.ndim == 3:
        mask = np.any(image != 0, axis=2).astype(np.uint8)
    else:
        mask = (image != 0).astype(np.uint8)
    return mask


def crop_to_nonzero(
    image: np.ndarray
) -> np.ndarray:
    """
    根據非零區域自動裁剪影像邊界。

    Args:
        image: 輸入影像
    Returns:
        cropped: 裁剪後影像
    """
    mask = create_mask_from_image(image)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]


def feather_mask(
    mask: np.ndarray,
    ksize: Tuple[int, int] = (31, 31)
) -> np.ndarray:
    """
    對二元遮罩進行高斯羽化，生成浮點型遮罩。

    Args:
        mask: 0/1 二元遮罩
        ksize: 高斯核大小，需為奇數
    Returns:
        feathered: [0.0,1.0] 浮點遮罩
    """
    mask_float = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_float, ksize, 0)
    max_val = blurred.max()
    if max_val > 0:
        blurred /= max_val
    return blurred


def pad_image(
    image: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_type: int = cv2.BORDER_CONSTANT,
    value: int = 0
) -> np.ndarray:
    """
    在影像邊界添加填充區域。

    Args:
        image: 輸入影像
        top, bottom, left, right: 分別為四邊填充像素數
        border_type: cv2 邊界類型
        value: 常數填充值，用於 CONSTANT 模式
    Returns:
        padded: 填充後影像
    """
    return cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=value)
