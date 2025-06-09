#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laplacian.py

功能：
  1. 讀取一張影像（jpg/png/…）
  2. 對灰階圖做二階微分（Laplacian）邊緣偵測
  3. 輸出邊緣圖為 JPEG

使用：
  pip install pillow
  python laplacian.py input.jpg output.jpg
  python laplacian.py /Users/young/Documents/nchu-2025-spring/DIP/hw3/input/img.png /Users/young/Documents/nchu-2025-spring/DIP/hw3/output/edge2.png
"""

import sys
from PIL import Image

# 3×3 Laplacian 四鄰域 Mask
LAPLACIAN_KERNEL = [
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
]

def normalize_to_255(edges, w, h):
    # 找出最大值
    M = max(max(row) for row in edges)
    if M == 0:
        return edges  # 整張都是 0，不用處理
    # 正規化
    for y in range(h):
        for x in range(w):
            edges[y][x] = int(edges[y][x] / M * 255)
    return edges

def image_to_pixels(img):
    """把 Pillow 灰階 Image 轉成 2D list"""
    w, h = img.size
    pix = img.load()
    return [[pix[x, y] for x in range(w)] for y in range(h)]

def laplacian_edge_detection(pixels, w, h, maxval=255):
    """
    對灰階像素做 Laplacian 卷積
    回傳同尺寸的邊緣強度 2D list
    """
    out = [[0]*w for _ in range(h)]
    pad = 1  # kernel 半徑
    for y in range(pad, h-pad):
        for x in range(pad, w-pad):
            s = 0
            for dy in range(-pad, pad+1):
                for dx in range(-pad, pad+1):
                    weight = LAPLACIAN_KERNEL[dy+pad][dx+pad]
                    s += weight * pixels[y+dy][x+dx]
            # 取絕對值，並截到 0~maxval
            val = abs(s)
            out[y][x] = min(maxval, val)
    return out

def pixels_to_image(pixels, w, h):
    """把 2D list 轉回 Pillow 灰階 Image"""
    img = Image.new('L', (w, h))
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), pixels[y][x])
    return img

def main():
    if len(sys.argv) != 3:
        print(f"用法: python {sys.argv[0]} input.jpg output.jpg")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    # 1. 讀影像並轉灰階
    img = Image.open(in_path).convert('L')
    w, h = img.size

    # 2. 灰階圖 → 2D list
    pixels = image_to_pixels(img)

    # 3. Laplacian 偵測
    edges = laplacian_edge_detection(pixels, w, h, maxval=255)

    edges = normalize_to_255(edges, w, h)

    # 4. 2D list → Image
    edge_img = pixels_to_image(edges, w, h)

    # 5. 存成 JPEG
    edge_img.save(out_path, format='JPEG', quality=95)
    print(f"Laplacian 邊緣偵測完成，結果存為：{out_path}")

if __name__ == '__main__':
    main()
