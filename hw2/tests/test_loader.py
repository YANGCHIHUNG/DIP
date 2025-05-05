import pytest
import numpy as np
import cv2
from src.loader import load_images

def test_load_images(tmp_path):
    # 建立一張 10×10 的黑色測試影像
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), img)

    # 使用 load_images 載入剛剛產生的影像
    imgs = load_images([str(img_path)])

    # 檢查回傳值型態與內容
    assert isinstance(imgs, list)
    assert len(imgs) == 1
    assert isinstance(imgs[0], np.ndarray)
    assert imgs[0].shape == img.shape
