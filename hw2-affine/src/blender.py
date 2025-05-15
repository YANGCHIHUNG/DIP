import cv2
import numpy as np


def warp_and_blend(
    base: np.ndarray,
    overlay: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """
    將 overlay 透視變形後融合到 base，簡單以重疊區域平均

    參數：
        base (np.ndarray): 基準影像 BGR 陣列
        overlay (np.ndarray): 待融合影像 BGR 陣列
        H (np.ndarray): 單應性矩陣

    回傳：
        result (np.ndarray): 融合後影像
    """
    h1, w1 = base.shape[:2]
    h2, w2 = overlay.shape[:2]

    # 計算 overlay 經透視轉換後的角點
    corners = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_corners = np.concatenate((
        np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2),
        warped_corners
    ), axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    trans = [-x_min, -y_min]

    # 平移矩陣
    H_trans = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]])

    # Warp overlay
    result = cv2.warpPerspective(overlay, H_trans.dot(H), (x_max-x_min, y_max-y_min))
    # 放置 base
    result[trans[1]:h1+trans[1], trans[0]:w1+trans[0]] = base

    # 簡單混合：重疊區域取平均
    return result