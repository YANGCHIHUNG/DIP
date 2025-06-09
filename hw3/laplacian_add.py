#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laplacian_add.py

功能：
  將原始彩色影像與二階微分邊緣圖直接相加，
  產生二階微分邊緣銳化影像。

使用：
  pip install pillow
  python laplacian_add.py \
      --original /Users/young/Documents/nchu-2025-spring/DIP/hw3/input/img.png \
      --edge2 /Users/young/Documents/nchu-2025-spring/DIP/hw3/output/edge2.png \
      --output /Users/young/Documents/nchu-2025-spring/DIP/hw3/output/sharp2.png

參數：
  --original  原始彩色影像路徑 (jpg/png/...)
  --edge2     二階微分邊緣圖 (灰階, 0~255)
  --output    輸出銳化後影像路徑
"""
import sys
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="將原圖與二階邊緣圖相加，生成銳化影像")
    parser.add_argument('--original', required=True, help='原始彩色影像路徑')
    parser.add_argument('--edge2', required=True, help='二階微分邊緣圖路徑')
    parser.add_argument('--output', required=True, help='輸出銳化後影像路徑')
    return parser.parse_args()

def load_images(orig_path, edge2_path):
    # 讀取原圖與邊緣圖
    orig = Image.open(orig_path).convert('RGB')
    edge = Image.open(edge2_path).convert('L')
    if orig.size != edge.size:
        raise ValueError('原始影像和邊緣圖尺寸必須相同')
    return orig, edge

def add_laplacian(orig, edge):
    w, h = orig.size
    orig_pix = orig.load()
    edge_pix = edge.load()

    # 建立輸出影像
    result = Image.new('RGB', (w, h))
    res_pix = result.load()

    for y in range(h):
        for x in range(w):
            r, g, b = orig_pix[x, y]
            e = edge_pix[x, y]
            # 加總並夾到 0-255
            nr = min(255, max(0, r + e))
            ng = min(255, max(0, g + e))
            nb = min(255, max(0, b + e))
            res_pix[x, y] = (nr, ng, nb)
    return result

def main():
    args = parse_args()
    orig, edge = load_images(args.original, args.edge2)
    result = add_laplacian(orig, edge)
    result.save(args.output, format='JPEG', quality=95)
    print(f'已輸出二階邊緣銳化影像：{args.output}')

if __name__ == '__main__':
    main()
