"""
Module: stitcher.py
功能：影像拼接主流程 (計算畫布範圍、仿射變換拼接、線性平均融合)
"""
from typing import List, Tuple
import cv2
import numpy as np


def compute_panorama_canvas(images: List[np.ndarray], Ms: List[np.ndarray]) -> Tuple[int, int, float, float]:
    # 計算拼接後畫布的寬、高與平移偏移量
    h0, w0 = images[0].shape[:2]
    corners = np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]], dtype=np.float32)
    all_corners = [corners]

    for M in Ms:
        transformed = cv2.transform(np.array([corners]), M)[0]
        all_corners.append(transformed)

    all_pts = np.vstack(all_corners)
    x_min, y_min = np.min(all_pts, axis=0)
    x_max, y_max = np.max(all_pts, axis=0)

    tx, ty = -x_min, -y_min
    width = int(np.ceil(x_max - x_min))
    height = int(np.ceil(y_max - y_min))
    return width, height, tx, ty


def warp_image(image: np.ndarray, M: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    # 使用仿射矩陣將影像 warp 到指定大小
    return cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR)


def simple_blend(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    # 線性平均融合：重疊區域取平均、非重疊區域直接覆蓋
    base_f = base.astype(np.float32)
    over_f = overlay.astype(np.float32)

    mask_base = np.any(base != 0, axis=2)
    mask_over = np.any(overlay != 0, axis=2)

    result = base_f.copy()
    only_over = mask_over & ~mask_base
    overlap = mask_over & mask_base

    # 非重疊新區域直接覆蓋
    result[only_over] = over_f[only_over]
    # 重疊區域取平均
    result[overlap] = (base_f[overlap] + over_f[overlap]) / 2

    return result.astype(np.uint8)


def stitch(images: List[np.ndarray], Ms: List[np.ndarray], do_blend: bool = True) -> np.ndarray:
    # 拼接多張影像為 panorama
    width, height, tx, ty = compute_panorama_canvas(images, Ms)
    size = (width, height)

    # 基準圖與偏移矩陣
    T = np.array([[1, 0, tx],
                  [0, 1, ty]], dtype=np.float32)
    panorama = warp_image(images[0], T, size)

    # 依序 warp 並融合後續影像
    for img, M in zip(images[1:], Ms):
        M_full = T.dot(np.vstack([M, [0, 0, 1]]))[:2]
        warped = warp_image(img, M_full, size)
        if do_blend:
            panorama = simple_blend(panorama, warped)
        else:
            panorama = warped

    return panorama
