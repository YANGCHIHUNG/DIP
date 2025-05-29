import cv2


def get_feature_detector(name='ORB', **kwargs):
    """
    根據名稱建立特徵偵測與描述子演算器。

    參數:
    - name (str): 特徵演算法名稱，可選 ORB/SIFT/AKAZE/KAZE/BRISK
    - kwargs: 傳給對應 cv2.create* 的參數

    回傳:
    - detector (cv2.Feature2D): 特徵偵測器與描述子物件
    """
    name = name.upper()
    if name == 'ORB':
        return cv2.ORB_create(**kwargs)
    elif name == 'SIFT':
        # SIFT 在 OpenCV-contrib 中
        return cv2.SIFT_create(**kwargs)
    elif name == 'AKAZE':
        return cv2.AKAZE_create(**kwargs)
    elif name == 'KAZE':
        return cv2.KAZE_create(**kwargs)
    elif name == 'BRISK':
        return cv2.BRISK_create(**kwargs)
    else:
        raise ValueError(f"Unknown feature detector: {name}")


def detect_and_compute(detector, image):
    """
    對影像執行關鍵點偵測與描述子計算。

    參數:
    - detector (cv2.Feature2D): 由 get_feature_detector 建立的偵測器
    - image (ndarray): 輸入影像，可為灰階或 BGR 彩色影像

    回傳:
    - keypoints (list of cv2.KeyPoint)
    - descriptors (ndarray): 描述子矩陣
    """
    # 若為彩色影像，先轉為灰階
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors
