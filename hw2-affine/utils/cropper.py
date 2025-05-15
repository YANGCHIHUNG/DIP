import cv2
import numpy as np
from typing import Any


def auto_crop(img: np.ndarray) -> np.ndarray:
    """
    自動裁剪影像中非零（有效）區域，去除黑邊

    參數:
        img (np.ndarray): 載入的 BGR 影像

    回傳:
        cropped (np.ndarray): 裁切後的影像
    """
    # 轉為灰階並二值化，找出有效像素
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    return cropped