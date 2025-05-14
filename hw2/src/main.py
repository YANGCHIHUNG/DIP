import argparse
import os
import yaml
import cv2
from loader import load_images
from stitcher import stitch_images

def parse_args():
    parser = argparse.ArgumentParser(
        description="Panorama Stitching Tool (Preserve Input Order)"
    )
    parser.add_argument(
        'images', nargs='+',
        help="依拍攝順序提供影像檔案路徑"
    )
    parser.add_argument(
        '--output', required=True,
        help="輸出全景圖檔案路徑（含副檔名）"
    )
    parser.add_argument(
        '--config', required=True,
        help="設定檔 yaml 路徑"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # 讀取設定檔
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 依照 CLI 提供的順序載入影像
    imgs = load_images(args.images)
    # 執行拼接
    panorama = stitch_images(imgs, cfg['ransac_threshold'])

    # 確保輸出目錄存在
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 統一副檔名
    base, _ = os.path.splitext(args.output)
    ext = cfg.get('output_format', 'jpg').lstrip('.')
    out_path = f"{base}.{ext}"

    # 寫出結果
    if not cv2.imwrite(out_path, panorama):
        raise IOError(f"無法將圖片寫入 {out_path}")
    print(f"全景圖已儲存：{out_path}")


if __name__ == "__main__":
    main()
