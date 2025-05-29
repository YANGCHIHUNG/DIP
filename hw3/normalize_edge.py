#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_edge.py

功能：
  1. 讀取一張灰階邊緣圖 (jpg/png/...)
  2. 做最小–最大正規化，將像素值映射到 0.0~1.0
  3. 可設定門檻 threshold，將小於 threshold 的弱邊緣設為 0
  4. 存成影像 (0~255) 或回傳權重矩陣 (浮點 0~1)

使用：
  pip install pillow
  python normalize_edge.py edge.png [weight.png] [threshold]
  python normalize_edge.py /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/blur_sobel.jpg /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/weight.jpg 0.2

參數：
  edge.png      輸入的邊緣圖 (灰階)
  weight.png    選填，輸出的權重圖 (灰階顯示 0~255)
  threshold     選填，門檻值 (0.0~1.0)，弱邊緣將被移除，預設 0.0

回傳：
  2D list of floats in [0.0,1.0]
"""
import sys
from PIL import Image

def normalize_edge_map(input_path, output_path=None, threshold=0.0):
    """
    讀入灰階邊緣圖，做 min-max 正規化到 [0.0,1.0]，
    並移除小於 threshold 的弱邊緣。
    若提供 output_path，則同時儲存顯示用的 0~255 灰階圖。
    回傳正規化的 2D list。
    """
    # 讀取並轉灰階
    img = Image.open(input_path).convert('L')
    w, h = img.size
    pix = img.load()

    # 找最小、最大值
    minv, maxv = 255, 0
    for y in range(h):
        for x in range(w):
            v = pix[x, y]
            if v < minv: minv = v
            if v > maxv: maxv = v
    denom = maxv - minv

    # 建權重矩陣
    weight_map = [[0.0]*w for _ in range(h)]

    # 全圖相同 or 無輸出需求
    if denom == 0 and output_path:
        Image.new('L', (w, h)).save(output_path)
        print(f"門檻化後全圖為零，已存全黑圖：{output_path}")
        return weight_map
    elif denom == 0:
        return weight_map

    # 建立顯示用影像
    if output_path:
        out_img = Image.new('L', (w, h))
        out_pix = out_img.load()

    # 計算正規化並套用門檻
    for y in range(h):
        for x in range(w):
            v = pix[x, y]
            norm = (v - minv) / denom
            # 門檻化：移除弱邊緣
            if norm < threshold:
                norm = 0.0
            weight_map[y][x] = norm
            if output_path:
                out_pix[x, y] = int(norm * 255 + 0.5)

    # 儲存顯示影像
    if output_path:
        out_img.save(output_path, format='PNG')
        print(f"已存權重圖 (0~255) 到：{output_path}, threshold={threshold}")

    return weight_map

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} input_edge.png [output_weight.png] [threshold]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    threshold = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.0

    weights = normalize_edge_map(inp, out, threshold)
    print("正規化並門檻化完成，回傳權重矩陣。")
