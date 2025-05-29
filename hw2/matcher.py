import cv2


def create_matcher(matcher_type='BF', norm_type=cv2.NORM_HAMMING, cross_check=False,
                   flann_index_params=None, flann_search_params=None):
    """
    建立特徵匹配器。

    參數:
    - matcher_type (str): 'BF' 或 'FLANN'
    - norm_type: BFMatcher 使用的距離度量，預設 NORM_HAMMING
    - cross_check (bool): BFMatcher 是否啟用交叉檢查
    - flann_index_params (dict): FLANN 的 index 參數
    - flann_search_params (dict): FLANN 的 search 參數

    回傳:
    - matcher (cv2.DescriptorMatcher)
    """
    if matcher_type.upper() == 'BF':
        return cv2.BFMatcher(norm_type, crossCheck=cross_check)
    elif matcher_type.upper() == 'FLANN':
        # FLANN 需要設定 index 與 search 參數
        if flann_index_params is None:
            # 1 為 KDTree
            flann_index_params = {'algorithm': 1, 'trees': 5}
        if flann_search_params is None:
            flann_search_params = {'checks': 50}
        return cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
    else:
        raise ValueError(f"Unknown matcher_type: {matcher_type}")


def match_descriptors(matcher, des1, des2, ratio_test=True, ratio=0.75, top_k=None):
    """
    執行描述子匹配，並可選用 Lowe's ratio test 和取前 K 筆。

    參數:
    - matcher: cv2.Matcher 物件
    - des1, des2: 兩組描述子
    - ratio_test (bool): 是否使用 ratio test
    - ratio (float): ratio test 閾值
    - top_k (int or None): 取最優前 K 筆匹配，None 表示不限制

    回傳:
    - matches (list of cv2.DMatch)
    """
    if ratio_test:
        # 使用 KNN 匹配 (k=2) 進行 ratio test
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in raw_matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        matches = sorted(good, key=lambda x: x.distance)
    else:
        # 直接匹配並依距離排序
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

    if top_k is not None and len(matches) > top_k:
        matches = matches[:top_k]
    return matches
