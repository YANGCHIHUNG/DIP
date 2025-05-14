import numpy as np
import cv2
from src.loader import load_images
from src.stitcher import stitch_images

def test_stitch_images_identical(tmp_path):
    """
    測試對兩張完全相同的影像進行拼接
    """
    # 建立兩張相同的空影像
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    path1 = tmp_path / "img1.jpg"
    path2 = tmp_path / "img2.jpg"
    cv2.imwrite(str(path1), img)
    cv2.imwrite(str(path2), img)

    # 載入並拼接
    imgs = load_images([str(path1), str(path2)])
    panorama = stitch_images(imgs, ransac_thresh=5.0)

    # 檢查拼接結果至少與原圖尺寸相同，且不為 None
    assert panorama is not None
    assert panorama.shape[0] >= img.shape[0]
    assert panorama.shape[1] >= img.shape[1]