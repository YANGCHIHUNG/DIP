import os
import cv2


def load_images_from_dir(directory, extensions=None):
    """
    從指定資料夾載入所有影像。

    參數:
    - directory (str): 影像資料夾路徑
    - extensions (list of str): 副檔名列表，預設 ['.jpg', '.jpeg', '.png', '.bmp']

    回傳:
    - images (dict): key 為檔名，value 為 cv2 讀取的影像陣列
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    if not os.path.isdir(directory):
        raise ValueError(f"'{directory}' 不是有效的資料夾路徑")

    images = {}
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in extensions:
            path = os.path.join(directory, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"警告: 無法讀取影像 {path}")
                continue
            images[fname] = img
    return images


def load_images(image_paths):
    """
    載入指定路徑列表中的影像。

    參數:
    - image_paths (list of str): 影像檔案完整路徑

    回傳:
    - images (dict): key 為檔名，value 為 cv2 讀取的影像陣列
    """
    images = {}
    for path in image_paths:
        if not os.path.isfile(path):
            raise ValueError(f"'{path}' 不是有效的檔案路徑")
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"無法載入影像: {path}")
        images[os.path.basename(path)] = img
    return images


def preprocess_image(img, to_gray=False, resize=None):
    """
    對影像進行前處理。

    參數:
    - img (ndarray): 原始影像
    - to_gray (bool): 是否轉為灰階，預設 False
    - resize (tuple of int): (width, height) 重新調整大小，預設 None

    回傳:
    - result (ndarray): 前處理後的影像
    """
    result = img.copy()
    if to_gray:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if resize is not None:
        result = cv2.resize(result, resize, interpolation=cv2.INTER_AREA)
    return result
