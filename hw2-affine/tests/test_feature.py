import numpy as np
import cv2
from src.feature import detect_and_compute

def test_detect_and_compute():
    # 建立 100×100 全黑影像，並畫一個白色方塊以產生特徵
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), (255, 255, 255), -1)

    # 偵測關鍵點與描述子
    kps, desc = detect_and_compute(img)

    # 檢查回傳型態與數量合理性
    assert isinstance(kps, list)
    assert len(kps) > 0, "應該至少偵測到一個關鍵點"
    assert desc is not None
    assert desc.shape[0] == len(kps), "描述子的數量應與關鍵點一致"
