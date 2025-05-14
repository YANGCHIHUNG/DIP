"""
Module: main.py
功能：整合 Panorama Affine Transform 流程
"""
import os
import yaml
import cv2
from typing import List

from loader import load_and_preprocess
from feature import extract_features
from matcher import create_bf_matcher, knn_match_descriptors, filter_matches_ratio, get_matched_points
from estimator import estimate_affine
from stitcher import stitch
from blender import multi_band_blend
from calibrator import load_calibration, undistort_image


def main(config_path: str):
    # 讀取參數設定
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 資料夾與參數
    folder = cfg['data_folder']
    max_w = cfg.get('max_width')
    max_h = cfg.get('max_height')
    to_gray = cfg.get('to_gray', False)
    blur = cfg.get('blur', True)
    matcher_params = cfg.get('matcher', {})
    detector_params = cfg.get('detector', {})
    ransac_thresh = cfg.get('ransac_thresh', 3.0)
    output_path = cfg['output_path']
    calib_file = cfg.get('calibration_file')
    do_blend = cfg.get('do_blend', False)
    blend_levels = cfg.get('blend_levels', 4)

    # 載入與前處理
    images, filenames = load_and_preprocess(folder, max_w, max_h, to_gray, blur)

    # 若有校正檔，先去畸變
    if calib_file and os.path.exists(calib_file):
        mtx, dist, _, _ = load_calibration(calib_file)
        images = [undistort_image(img, mtx, dist) for img in images]

    # 特徵偵測與匹配
    keypoints_list = []
    descriptors_list = []
    for img in images:
        method = detector_params.get('method', 'ORB')
        params = {k: v for k, v in detector_params.items() if k != 'method'}
        kp, desc = extract_features(img,
                                    method=method,
                                    detector_params=params)
        keypoints_list.append(kp)
        descriptors_list.append(desc)

    # 建立 Matcher
    matcher = create_bf_matcher(**matcher_params)

    # 計算仿射矩陣
    Ms = []
    for i in range(len(images) - 1):
        matches_knn = knn_match_descriptors(matcher, descriptors_list[i], descriptors_list[i+1])
        good = filter_matches_ratio(matches_knn, cfg.get('ratio_test', 0.75))
        pts1, pts2 = get_matched_points(keypoints_list[i], keypoints_list[i+1], good)
        M, mask = estimate_affine(pts1, pts2, method=cfg.get('est_method', 'partial'), ransac_thresh=ransac_thresh)
        Ms.append(M)

    # 拼接
    panorama = stitch(images, Ms, do_blend=True)

    # 若選用多頻段融合，與最後一張做融合
    if do_blend and len(images) > 1:
        mask = cv2.cvtColor((panorama > 0).astype('uint8')*255, cv2.COLOR_BGR2GRAY) / 255.0
        panorama = multi_band_blend(panorama, panorama, mask, levels=blend_levels)

    # 儲存結果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Panorama Affine Transform Pipeline')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='配置檔路徑')
    args = parser.parse_args()
    main(args.config)
