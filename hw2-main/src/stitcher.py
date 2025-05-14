import numpy as np
from typing import List
from .feature import detect_and_compute
from .matcher import match_features
from .estimator import find_homography
from .blender import warp_and_blend


def stitch_images(
    imgs: List[np.ndarray],
    ransac_thresh: float
) -> np.ndarray:
    """
    迭代將多張影像拼接成全景圖，若匹配條件不符則略過該張影像
    """
    result = imgs[0]
    for img in imgs[1:]:
        kps1, desc1 = detect_and_compute(result)
        kps2, desc2 = detect_and_compute(img)
        # descriptor 不可為 None
        if desc1 is None or desc2 is None:
            continue
        matches = match_features(desc1, desc2)
        # 匹配點不足時跳過
        if len(matches) < 4:
            continue
        H = find_homography(kps1, kps2, matches, ransac_thresh)
        if H is None:
            continue
        result = warp_and_blend(result, img, H)
    return result