#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laplacian_sharpen.py

功能：
  1. 讀取一張影像 (jpg/png/...)
  2. 對灰階圖做二階微分 (Laplacian)，並做最小-最大正規化
  3. 將邊緣結果加回到彩色原圖 (RGB 三通道銳化)
  4. 輸出銳化後影像為 JPEG

使用：
  pip install pillow
  python laplacian_sharpen.py input.jpg output.jpg [alpha]

  python laplacian_sharpen.py /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/input/img.png /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/laplacian_sharpen_img.jpg 1.5

參數：
  input.jpg   原始影像路徑
  output.jpg  銳化後輸出影像路徑
  alpha       邊緣加回強度 (預設 1.0)
"""
import sys
import math
from PIL import Image

# 3×3 Laplacian 四鄰域 Mask
LAPLACIAN_KERNEL = [
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
]

def image_to_pixels(img):
    """把 Pillow 灰階 Image 轉成 2D list"""
    w, h = img.size
    pix = img.load()
    return [[pix[x, y] for x in range(w)] for y in range(h)]

def laplacian_edge_detection(pixels, w, h):
    """對灰階像素做 Laplacian 卷積，回傳 2D list"""
    out = [[0]*w for _ in range(h)]
    pad = 1
    for y in range(pad, h-pad):
        for x in range(pad, w-pad):
            s = 0
            for dy in range(-pad, pad+1):
                for dx in range(-pad, pad+1):
                    weight = LAPLACIAN_KERNEL[dy+pad][dx+pad]
                    s += weight * pixels[y+dy][x+dx]
            out[y][x] = abs(s)
    return out

def normalize_to_255(edges, w, h):
    """將邊緣強度正規化到 0~255"""
    M = max(max(row) for row in edges)
    if M == 0:
        return edges
    for y in range(h):
        for x in range(w):
            edges[y][x] = int(edges[y][x] / M * 255)
    return edges


def sharpen_color_image(color_img, edges, w, h, alpha):
    """將正規化後邊緣加回原彩色影像 (RGB)，回傳 W×H 影像"""
    out_img = Image.new('RGB', (w, h))
    orig_pix = color_img.load()
    out_pix = out_img.load()
    for y in range(h):
        for x in range(w):
            r, g, b = orig_pix[x, y]
            e = edges[y][x]
            # 對每個通道分別加強
            nr = min(255, max(0, int(r + alpha * e)))
            ng = min(255, max(0, int(g + alpha * e)))
            nb = min(255, max(0, int(b + alpha * e)))
            out_pix[x, y] = (nr, ng, nb)
    return out_img


def main():
    if len(sys.argv) < 3:
        print(f"用法: python {sys.argv[0]} input.jpg output.jpg [alpha]")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]
    alpha    = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0

    # 1. 讀影像 (RGB) 及灰階版本
    color_img = Image.open(in_path).convert('RGB')
    gray_img  = color_img.convert('L')
    w, h = gray_img.size

    # 2. 灰階轉成 2D list
    gray_pixels = image_to_pixels(gray_img)

    # 3. Laplacian 偵測
    edges = laplacian_edge_detection(gray_pixels, w, h)

    # 4. 正規化
    edges = normalize_to_255(edges, w, h)

    # 5. 銳化彩色影像
    result_img = sharpen_color_image(color_img, edges, w, h, alpha)

    # 6. 儲存結果
    result_img.save(out_path, format='JPEG', quality=95)
    print(f"已完成 Laplacian 彩色銳化，結果存為：{out_path} (alpha={alpha})")

if __name__ == '__main__':
    main()