"""
Module: calibrator.py
功能：相機 / 環境校正 (Camera Calibration)
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


def calibrate_camera(
    image_paths: List[str],
    chessboard_size: Tuple[int, int],
    square_size: float = 1.0,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    使用棋盤格圖進行相機校正，並可選擇儲存結果

    Args:
        image_paths: 校正影像檔案路徑列表
        chessboard_size: 內角點 (cols, rows)
        square_size: 格子邊長（單位自行定義，如公分）
        save_path: 若提供路徑，則以 .npz 格式儲存校正參數

    Returns:
        camera_matrix: 相機內參矩陣
        dist_coeffs: 畸變參數
        rvecs: 旋轉向量列表
        tvecs: 平移向量列表
    """
    # 準備世界座標下的棋盤格點
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints: List[np.ndarray] = []  # 世界座標點
    imgpoints: List[np.ndarray] = []  # 影像座標點
    img_shape = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
            )
            imgpoints.append(corners2)

    if not objpoints:
        raise ValueError("未找到任何棋盤格角點，請檢查影像路徑和棋盤格設定。")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    if save_path:
        np.savez(save_path,
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs,
                 rvecs=rvecs,
                 tvecs=tvecs)

    return camera_matrix, dist_coeffs, rvecs, tvecs


def load_calibration(
    calib_file: str
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    從 .npz 檔載入校正參數

    Returns:
        camera_matrix, dist_coeffs, rvecs, tvecs
    """
    data = np.load(calib_file, allow_pickle=True)
    return data['camera_matrix'], data['dist_coeffs'], data['rvecs'].tolist(), data['tvecs'].tolist()


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    new_camera_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    對影像應用去畸變

    Args:
        image: 原始影像
        camera_matrix: 相機內參
        dist_coeffs: 畸變參數
        new_camera_matrix: 若提供，作為最佳內參

    Returns:
        去畸變後影像
    """
    h, w = image.shape[:2]
    if new_camera_matrix is None:
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted
