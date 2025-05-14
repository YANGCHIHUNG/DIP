"""
Module: loader.py
功能：影像讀取與前處理 (resizing, grayscale, blur)
"""
import os
from typing import List, Tuple, Optional
import cv2
import numpy as np

def load_images_from_folder(
    folder: str,
    exts: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
) -> Tuple[List[np.ndarray], List[str]]:
    """
    從指定資料夾載入所有圖片。

    Args:
        folder: 圖片資料夾路徑
        exts: 支援的檔案副檔名列表

    Returns:
        images: 讀取到的 BGR 影像列表
        filenames: 對應的檔名列表
    """
    images: List[np.ndarray] = []
    filenames: List[str] = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in exts:
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            images.append(img)
            filenames.append(fname)
    return images, filenames


def resize_image(
    image: np.ndarray,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None
) -> np.ndarray:
    """
    根據最大寬高縮放影像，保持長寬比。

    Args:
        image: 原始影像
        max_width: 最大寬度 (px)，若 None 不做水平限制
        max_height: 最大高度 (px)，若 None 不做垂直限制

    Returns:
        縮放後影像 (或原影像)
    """
    h, w = image.shape[:2]
    scale = 1.0
    if max_width is not None and w > max_width:
        scale = min(scale, max_width / w)
    if max_height is not None and h > max_height:
        scale = min(scale, max_height / h)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def preprocess_image(
    image: np.ndarray,
    to_gray: bool = False,
    blur: bool = True,
    blur_ksize: Tuple[int, int] = (5, 5)
) -> np.ndarray:
    """
    對影像進行可選的灰階化與高斯模糊。

    Args:
        image: 原始影像或已縮放影像
        to_gray: 是否轉為灰階
        blur: 是否套用 Gaussian Blur
        blur_ksize: 模糊核大小

    Returns:
        處理後影像
    """
    processed = image.copy()
    if to_gray:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if blur:
        processed = cv2.GaussianBlur(processed, blur_ksize, 0)
    return processed


def load_and_preprocess(
    folder: str,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    to_gray: bool = False,
    blur: bool = True
) -> Tuple[List[np.ndarray], List[str]]:
    """
    結合載入與前處理：讀取資料夾內所有圖片，並依序執行 resize 與 preprocess。

    Args:
        folder: 圖片資料夾路徑
        max_width: 縮放最大寬度
        max_height: 縮放最大高度
        to_gray: 是否灰階化
        blur: 是否高斯模糊

    Returns:
        processed_images: 處理後影像列表
        filenames: 原始檔名列表
    """
    images, filenames = load_images_from_folder(folder)
    processed_images: List[np.ndarray] = []
    for img in images:
        img_resized = resize_image(img, max_width, max_height)
        img_pre = preprocess_image(img_resized, to_gray, blur)
        processed_images.append(img_pre)
    return processed_images, filenames
