import cv2
from typing import List


def load_images(paths: List[str]) -> List:
    """
    載入多張影像，返回 BGR 形式的 numpy 陣列清單

    參數：
        paths (List[str]): 圖片檔案路徑列表

    回傳：
        images (List[np.ndarray]): 載入後的圖像列表
    """
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"無法讀取影像：{p}")
        images.append(img)
    return images