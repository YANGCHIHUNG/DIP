import cv2
import numpy as np

def create_weight_map(mask, blur_radius=21):
    """
    建立羽化權重圖（Feathering Weight Map），以平滑邊界。

    參數:
    - mask (ndarray): 二值遮罩，前景為 255
    - blur_radius (int): 高斯模糊半徑，須為奇數

    回傳:
    - weight_map (ndarray): float32, 範圍 [0,1]
    """
    # 將遮罩轉為浮點
    mask_f = mask.astype(np.float32) / 255.0
    # 使用高斯模糊產生平滑過渡
    weight = cv2.GaussianBlur(mask_f, (blur_radius, blur_radius), 0)
    # 標準化至 [0,1]
    weight_map = weight / (weight.max() + 1e-8)
    return weight_map


def feather_blend(base_img, warped_img, mask, blur_radius=21):
    """
    以羽化方式混合兩張影像。

    參數:
    - base_img (ndarray): 基準影像
    - warped_img (ndarray): 經仿射變換後的影像
    - mask (ndarray): warped_img 的二值遮罩
    - blur_radius (int): 權重圖高斯模糊半徑

    回傳:
    - blended (ndarray): 混合後的影像
    """
    # 建立羽化權重圖
    w = create_weight_map(mask, blur_radius)[..., np.newaxis]
    # 轉為浮點數進行加權
    base = base_img.astype(np.float32)
    warp = warped_img.astype(np.float32)
    # alpha 混合
    blended = base * (1 - w) + warp * w
    return blended.astype(np.uint8)


def multiband_blend(img1, img2, mask, num_levels=5):
    """
    使用多頻帶金字塔（Laplacian Pyramid）進行混合。

    參數:
    - img1, img2 (ndarray): 兩張要混合的 BGR 影像
    - mask (ndarray): 混合遮罩，值為 0/1
    - num_levels (int): 金字塔層數

    回傳:
    - blended (ndarray): 混合後的影像
    """
    # 建立 Gaussian Pyramid
    G1 = img1.copy().astype(np.float32)
    G2 = img2.copy().astype(np.float32)
    GM = mask.copy().astype(np.float32)
    gp1 = [G1]
    gp2 = [G2]
    gpm = [GM]
    for i in range(num_levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        GM = cv2.pyrDown(GM)
        gp1.append(G1)
        gp2.append(G2)
        gpm.append(GM)

    # 建立 Laplacian Pyramid
    lp1 = [gp1[-1]]
    lp2 = [gp2[-1]]
    for i in range(num_levels, 0, -1):
        size = (gp1[i-1].shape[1], gp1[i-1].shape[0])
        L1 = cv2.subtract(gp1[i-1], cv2.pyrUp(gp1[i], dstsize=size))
        L2 = cv2.subtract(gp2[i-1], cv2.pyrUp(gp2[i], dstsize=size))
        lp1.append(L1)
        lp2.append(L2)

    # 混合金字塔
    LS = []
    for l1, l2, gm in zip(lp1, lp2, gpm[::-1]):
        gm_exp = cv2.merge([gm, gm, gm])
        ls = l1 * (1 - gm_exp) + l2 * gm_exp
        LS.append(ls)

    # 還原影像
    blended = LS[0]
    for i in range(1, len(LS)):
        size = (LS[i].shape[1], LS[i].shape[0])
        blended = cv2.pyrUp(blended, dstsize=size)
        blended = cv2.add(blended, LS[i])

    return np.clip(blended, 0, 255).astype(np.uint8)


def blend_images(base_img, warped_img, mask, method='feather', **kwargs):
    """
    通用混合介面。

    參數:
    - base_img, warped_img (ndarray): 要混合的影像
    - mask (ndarray): 二值遮罩，255 表示 warped_img 區域
    - method (str): 'feather' 或 'multiband'
    - kwargs: 傳給對應混合方法的參數

    回傳:
    - blended (ndarray)
    """
    method = method.lower()
    if method == 'feather':
        return feather_blend(base_img, warped_img, mask, **kwargs)
    elif method == 'multiband':
        # 將 mask 轉為 0/1
        m = (mask.astype(np.float32) / 255.0)
        return multiband_blend(base_img, warped_img, m, **kwargs)
    else:
        raise ValueError(f"Unknown blend method: {method}")
