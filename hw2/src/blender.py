import numpy as np

class Blender:
    """
    簡單融合器：對應像素取最大值，保留任何一張圖的亮部細節。
    """
    def __init__(self):
        pass

    def blend(self, base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        # 直接對應位置取最大值，black(0) 的地方會被 overlay 的白色(255) 替代
        return np.maximum(base, overlay)
