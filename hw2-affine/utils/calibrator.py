# utils/calibrator.py
import cv2
import numpy as np
from typing import List, Tuple


def calibrate_camera(
    images: List[np.ndarray],
    pattern_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    使用多張棋盤格影像標定相機，並回傳相機參數與失真係數。

    參數:
        images (List[np.ndarray]): 載入的 BGR 圖像列表
        pattern_size (Tuple[int,int]): 棋盤格內角點行列數 (columns, rows)

    回傳:
        mtx (np.ndarray): 3x3 相機內參矩陣
        dist (np.ndarray): 失真係數向量 [k1, k2, p1, p2, k3]
        rvecs (List[np.ndarray]): 各張影像的旋轉向量
        tvecs (List[np.ndarray]): 各張影像的平移向量
    """
    # 準備物體座標 (0,0,0), (1,0,0), ..., (columns-1,rows-1,0)
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    objpoints: List[np.ndarray] = []  # 三維世界座標
    imgpoints: List[np.ndarray] = []  # 二維影像座標

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            continue
        # 提升角點精度
        corners2 = cv2.cornerSubPix(
            gray, corners, winSize=(11,11), zeroZone=(-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners2)

    if len(objpoints) < 1:
        raise ValueError("找不到足夠的棋盤格角點，請確認圖像與 pattern_size 設定正確。")

    # 執行相機標定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return mtx, dist, rvecs, tvecs
