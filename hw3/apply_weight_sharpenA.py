#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_weight_sharpenA.py

功能：
  根據一階邊緣權重圖，在邊緣強的地方將二階銳化後影像與原始影像進行融合，
  其餘區域維持原始影像不變。

使用：
  pip install pillow
  python apply_weight_sharpenA.py \
      --original orig.jpg \
      --sharpen sharpened.jpg \
      --weight weight.png \
      --output result.jpg

  python apply_weight_sharpenA.py \
      --original /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/input/img.png \
      --sharpen /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/laplacian_sharpen_img.jpg \
      --weight /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/weight.png \
      --output /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/img_resultA.png
參數：
  --original   原始彩色影像路徑
  --sharpen    二階銳化後的彩色影像路徑
  --weight     一階邊緣權重圖 (灰階, 0~255 表示 0.0~1.0)
  --output     最終輸出影像路徑
"""
import sys
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="融合一階邊緣權重與二階銳化影像")
    parser.add_argument('--original', required=True, help='原始彩色影像')
    parser.add_argument('--sharpen', required=True, help='二階銳化後彩色影像')
    parser.add_argument('--weight',   required=True, help='一階邊緣權重圖 (灰階)')
    parser.add_argument('--output',   required=True, help='輸出結果影像')
    return parser.parse_args()

def load_images(orig_path, sharp_path, weight_path):
    # 讀取影像
    orig = Image.open(orig_path).convert('RGB')
    sharp = Image.open(sharp_path).convert('RGB')
    wmap = Image.open(weight_path).convert('L')
    if orig.size != sharp.size or orig.size != wmap.size:
        raise ValueError('三張影像尺寸必須相同')
    return orig, sharp, wmap


def apply_weight_fusion(orig, sharp, wmap):
    """
    將 sharpen 與 orig 依照 wmap 權重融合：
      result = orig * (1 - w) + sharp * w
    其中 w = wmap[x,y] / 255.0
    """
    w, h = orig.size
    orig_pix = orig.load()
    sharp_pix = sharp.load()
    wmap_pix = wmap.load()

    result = Image.new('RGB', (w, h))
    res_pix = result.load()

    for y in range(h):
        for x in range(w):
            w_val = wmap_pix[x, y] / 255.0
            r0, g0, b0 = orig_pix[x, y]
            r1, g1, b1 = sharp_pix[x, y]
            r = int(r0 * (1 - w_val) + r1 * w_val)
            g = int(g0 * (1 - w_val) + g1 * w_val)
            b = int(b0 * (1 - w_val) + b1 * w_val)
            res_pix[x, y] = (r, g, b)
    return result


def main():
    args = parse_args()
    orig, sharp, wmap = load_images(args.original, args.sharpen, args.weight)
    result = apply_weight_fusion(orig, sharp, wmap)
    result.save(args.output, format='JPEG', quality=95)
    print(f'已完成加權融合，結果存為：{args.output}')

if __name__ == '__main__':
    main()
