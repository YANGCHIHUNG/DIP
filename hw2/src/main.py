# src/main.py
import argparse
import os
import yaml
import cv2
from src.loader import load_images
from src.stitcher import stitch_images

def parse_args():
    parser = argparse.ArgumentParser(description="Panorama Stitching Tool")
    parser.add_argument("--input",  required=True, help="輸入影像資料夾路徑")
    parser.add_argument("--output", required=True, help="輸出全景圖檔案路徑（含副檔名可省略）")
    parser.add_argument("--config", required=True, help="設定檔 yaml 路徑")
    return parser.parse_args()

def main():
    args = parse_args()
    # 讀設定檔
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 收集所有影像路徑
    img_files = sorted([
        os.path.join(args.input, fn)
        for fn in os.listdir(args.input)
        if fn.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    if not img_files:
        raise ValueError(f"在資料夾 {args.input} 中找不到任何影像檔。")

    # 載入影像並拼接
    imgs = load_images(img_files)
    panorama = stitch_images(imgs, cfg['ransac_threshold'])

    # 確保輸出目錄存在
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 統一副檔名
    base, _ = os.path.splitext(args.output)
    ext = cfg.get('output_format', 'jpg').lstrip('.')
    out_path = f"{base}.{ext}"

    # 寫出檔案
    success = cv2.imwrite(out_path, panorama)
    if not success:
        raise IOError(f"無法將圖片寫入 {out_path}")

    print(f"全景圖已儲存：{out_path}")

if __name__ == "__main__":
    main()
