import cv2
import numpy as np

def estimate_affine_ransac(src_pts, dst_pts, ransac_thresh=5.0, max_iters=2000):
    """
    使用 RANSAC 從 src_pts → dst_pts 估算 2x3 仿射矩陣。
    回傳:
      M   : (2,3) 仿射矩陣
      mask: (N,) 內點標記 1/0
    """
    N = src_pts.shape[0]
    if N < 3:
        raise ValueError("至少需要 3 對點才能估算仿射變換。")
    
    best_M = None
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0

    # 將點加上常數項，方便一次求解
    def solve_affine(p_src, p_dst):
        # p_src, p_dst shape=(k,2)，k>=3
        # 方程: [ x y 1 0 0 0 ] [a]   [x']
        #         [ 0 0 0 x y 1 ] [b] = [y']
        A = []
        b = []
        for (x, y), (xp, yp) in zip(p_src, p_dst):
            A.append([x, y, 1, 0, 0, 0])
            A.append([0, 0, 0, x, y, 1])
            b.append(xp)
            b.append(yp)
        A = np.array(A)    # shape=(2k,6)
        b = np.array(b)    # shape=(2k,)
        # 最小二乘解
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        # x = [a, b, tx, c, d, ty]
        return np.array([[x[0], x[1], x[2]],
                         [x[3], x[4], x[5]]])

    for i in range(max_iters):
        # 隨機抽 3 對點
        idx = np.random.choice(N, size=3, replace=False)
        M_candidate = solve_affine(src_pts[idx], dst_pts[idx])

        # 計算所有點的重投影誤差
        src_h = np.hstack([src_pts, np.ones((N,1))])       # shape=(N,3)
        proj = (M_candidate @ src_h.T).T                   # shape=(N,2)
        errs = np.linalg.norm(proj - dst_pts, axis=1)      # shape=(N,)

        # 以閾值分類內點
        inliers = errs < ransac_thresh
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_M = M_candidate

            # 若已經大部分皆為內點，就可提早停止
            if count > 0.8 * N:
                break

    if best_M is None:
        raise RuntimeError("RANSAC 無法估算出有效模型。")

    # 用所有內點再做一次最小二乘擬合以精緻化矩陣
    best_M = solve_affine(src_pts[best_inliers], dst_pts[best_inliers])

    # 回傳 M 與內點遮罩（1/0）
    return best_M, best_inliers.astype(np.uint8)


def estimate_affine_transform(kp1, kp2, matches, ransac_thresh=5.0, max_iters=2000):
    """
    根據匹配的關鍵點估算仿射變換矩陣。

    參數:
    - kp1 (list of cv2.KeyPoint): 參考影像的關鍵點
    - kp2 (list of cv2.KeyPoint): 待變換影像的關鍵點
    - matches (list of cv2.DMatch): 兩影像之間的匹配列表
    - ransac_thresh (float): RANSAC 重投影閾值，預設 5.0
    - max_iters (int): RANSAC 最大迭代次數，預設 2000

    回傳:
    - M (ndarray of shape (2,3)): 估算出的仿射矩陣
    - mask (ndarray): 表示內點(inliers)的遮罩
    """
    # 構造點陣
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    '''
    # 使用 RANSAC 估算仿射矩陣
    M, mask = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=max_iters
    )
    '''

    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    M, mask = estimate_affine_ransac(src, dst,
                                    ransac_thresh=ransac_thresh,
                                    max_iters=max_iters)

    return M, mask



def warp_image(img, M, output_shape, flags=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    """
    對影像進行仿射變換。

    參數:
    - img (ndarray): 輸入影像
    - M (ndarray of shape (2,3)): 仿射矩陣
    - output_shape (tuple of int): (width, height)
    - flags: OpenCV 插值標誌，預設雙線性
    - border_mode: 邊界模式，預設填充常數
    - border_value: 邊界填充值

    回傳:
    - warped (ndarray): 轉換後的影像
    """
    warped = cv2.warpAffine(
        img,
        M,
        output_shape,
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value
    )
    return warped


def compose_transforms(transforms):
    """
    將多個仿射矩陣依序相乘，生成合成矩陣。

    參數:
    - transforms (list of ndarray): 仿射矩陣列表，如 [M1, M2, ...]

    回傳:
    - M_comp (ndarray): 合成後的仿射矩陣
    """
    # 將 2x3 轉為 3x3 同構矩陣進行連乘
    M_comp = np.eye(3)
    for M in transforms:
        M_h = np.vstack([M, [0, 0, 1]])
        M_comp = M_h @ M_comp
    return M_comp[:2, :]
