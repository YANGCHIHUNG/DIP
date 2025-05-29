import os
import cv2
import yaml
import numpy as np
import argparse

from loader import load_images_from_dir, preprocess_image
from feature import get_feature_detector, detect_and_compute
from matcher import create_matcher, match_descriptors
from transformer import estimate_affine_transform
from blender import blend_images


def stitch_images(config_path, input_dir, output_path):
    # 讀取設定檔
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 建立特徵偵測器
    feat_cfg = cfg.get('feature', {})
    detector = get_feature_detector(
        feat_cfg.get('type', 'ORB'),
        **feat_cfg.get('params', {})
    )

    # 建立描述子匹配器
    matcher_cfg = cfg.get('matcher', {})
    norm_type = getattr(cv2, matcher_cfg.get('params', {}).get('norm_type', 'NORM_HAMMING'))
    cross_check = matcher_cfg.get('params', {}).get('cross_check', False)
    matcher = create_matcher(
        matcher_cfg.get('type', 'BF'),
        norm_type,
        cross_check,
        **matcher_cfg.get('flann', {})
    )

    # 其他參數
    match_cfg = cfg.get('match', {})
    ransac_cfg = cfg.get('ransac', {})
    blend_cfg = cfg.get('blend', {})
    blend_method = blend_cfg.get('method', 'feather')
    blend_params = blend_cfg.get('params', {})

    # 載入影像
    images_dict = load_images_from_dir(input_dir)
    if len(images_dict) < 2:
        raise ValueError("至少需要兩張影像進行拼接。")
    keys = sorted(images_dict.keys())
    imgs = [images_dict[k] for k in keys]

    # 預先偵測第一張影像特徵
    gray_prev = preprocess_image(imgs[0], to_gray=True)
    kp_prev, des_prev = detect_and_compute(detector, gray_prev)

    # 儲存每張影像至基準影像的合成仿射矩陣 (3x3)
    transforms = [np.eye(3)]

    # 依序估算仿射矩陣並串接
    for idx in range(1, len(imgs)):
        gray_cur = preprocess_image(imgs[idx], to_gray=True)
        kp_cur, des_cur = detect_and_compute(detector, gray_cur)

        # 匹配描述子
        matches = match_descriptors(
            matcher, des_prev, des_cur,
            ratio_test=match_cfg.get('ratio_test', True),
            ratio=match_cfg.get('ratio', 0.75),
            top_k=match_cfg.get('top_k', None)
        )
        if len(matches) < 3:
            raise RuntimeError(f"影像 '{keys[idx-1]}' 與 '{keys[idx]}' 匹配點不足：{len(matches)} < 3")

        # 估算仿射矩陣 (2x3)
        M23, inlier_mask = estimate_affine_transform(
            kp_prev, kp_cur, matches,
            ransac_thresh=ransac_cfg.get('thresh', 5.0),
            max_iters=ransac_cfg.get('max_iters', 2000)
        )
        if M23 is None:
            raise RuntimeError(f"影像 '{keys[idx-1]}' 與 '{keys[idx]}' 仿射估算失敗。")

        # 轉為 3x3
        M3 = np.vstack([M23, [0, 0, 1]])
        # 串接到第一張影像坐標系
        transforms.append(M3 @ transforms[-1])

        # 更新基準特徵
        kp_prev, des_prev = kp_cur, des_cur

    # 計算所有影像投影後的外框
    all_corners = []
    for img, T in zip(imgs, transforms):
        h, w = img.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        proj = T @ corners
        proj = proj[:2] / proj[2]
        all_corners.append(proj.T)
    all_pts = np.vstack(all_corners)

    # 全域邊界
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)

    # 平移到正值座標
    offset = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))

    # 建立空白畫布
    panorama = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    panorama_mask = np.zeros((out_h, out_w), dtype=np.uint8)

    # 依序 Warp 與混合
    for idx, img in enumerate(imgs):
        T = offset @ transforms[idx]
        M = T[:2]
        warp = cv2.warpAffine(
            img, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        mask = cv2.warpAffine(
            np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255,
            M, (out_w, out_h)
        )
        if idx == 0:
            panorama = warp
            panorama_mask = mask
        else:
            panorama = blend_images(
                panorama, warp, mask,
                method=blend_method,
                **blend_params
            )
            # 更新 mask
            panorama_mask = np.where(mask > 0, 255, panorama_mask).astype(np.uint8)

    # 儲存結果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, panorama)
    print(f"拼接完成，結果儲存至：{output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多張影像全景拼接 (Affine + Blend)')
    parser.add_argument('--config', required=True, help='設定檔路徑 (YAML)')
    parser.add_argument('--input', required=True, help='輸入影像資料夾')
    parser.add_argument('--output', required=True, help='輸出拼接結果路徑')
    args = parser.parse_args()
    stitch_images(args.config, args.input, args.output)