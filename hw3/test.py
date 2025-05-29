#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_weight_map.py

功能：
  將原始彩色影像每個像素乘上對應位置的一階邊緣權重，
  只保留邊緣區域的色彩，平坦區域隨權重變暗。

使用：
  pip install pillow
  python apply_weight_map.py \
      --input orig.jpg \
      --weight weight.png \
      --output result.jpg

  python test.py \
      --input /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/input/img.png \
      --weight /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/weight.png \
      --output /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/test.png


參數：
  --input    原始彩色影像路徑
  --weight   一階邊緣權重圖 (灰階, 0~255 對應 0.0~1.0)
  --output   輸出結果影像
"""
import sys
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="運用權重圖遮罩原圖，只保留邊緣色彩")
    parser.add_argument('--input', required=True, help='原始彩色影像路徑')
    parser.add_argument('--weight', required=True, help='一階邊緣權重圖 (灰階)')
    parser.add_argument('--output', required=True, help='輸出結果影像路徑')
    return parser.parse_args()

def load_images(input_path, weight_path):
    img = Image.open(input_path).convert('RGB')
    weight = Image.open(weight_path).convert('L')
    if img.size != weight.size:
        raise ValueError('原圖與權重圖尺寸必須相同')
    return img, weight

def apply_weight_mask(img, weight):
    w, h = img.size
    img_pix = img.load()
    weight_pix = weight.load()

    result = Image.new('RGB', (w, h))
    res_pix = result.load()

    for y in range(h):
        for x in range(w):
            # 權重映射到 [0.0,1.0]
            w_val = weight_pix[x, y] / 255.0
            r, g, b = img_pix[x, y]
            # 按權重乘色彩
            nr = int(r * w_val)
            ng = int(g * w_val)
            nb = int(b * w_val)
            res_pix[x, y] = (nr, ng, nb)
    return result

def main():
    args = parse_args()
    img, weight = load_images(args.input, args.weight)
    result = apply_weight_mask(img, weight)
    result.save(args.output, format='JPEG', quality=95)
    print(f'結果已儲存至：{args.output}')

if __name__ == '__main__':
    main()
