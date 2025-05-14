#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaussian_blur.py

功能：
  1. 讀取一張 JPEG 影像（使用 Pillow）
  2. 以純 Python 生成 Gaussian kernel
  3. 對影像做卷積模糊（RGB 三通道）
  4. 將結果輸出為 JPEG

使用：
  pip install pillow
  python gaussian_blur.py input.jpg output.jpg --ksize 5 --sigma 1.0
  python gaussian_blur.py /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/input/img.png /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/blur_img.png --ksize 5 --sigma 1.0
"""

import sys
import math
from PIL import Image

def make_gaussian_kernel(ksize: int, sigma: float):
    """
    生成 ksize×ksize 的 Gaussian kernel
    ksize: 核大小（必須為奇數）
    sigma: 標準差
    回傳一個 list of list，且已經正規化（和為 1）
    """
    assert ksize % 2 == 1, "ksize 必須為奇數"
    center = ksize // 2
    kernel = [[0.0]*ksize for _ in range(ksize)]
    sum_val = 0.0

    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            kernel[i][j] = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
            sum_val += kernel[i][j]

    # 正規化
    for i in range(ksize):
        for j in range(ksize):
            kernel[i][j] /= sum_val

    return kernel

def apply_gaussian_blur(img: Image.Image, kernel):
    """
    對 PIL RGB 影像做 Gaussian 卷積
    回傳新的 PIL Image
    """
    w, h = img.size
    pixels = img.load()
    # 準備輸出影像
    out = Image.new('RGB', (w, h))
    out_pix = out.load()

    ksize = len(kernel)
    pad = ksize // 2

    # 對每個像素做卷積
    for y in range(h):
        for x in range(w):
            r_sum = g_sum = b_sum = 0.0
            for dy in range(-pad, pad+1):
                for dx in range(-pad, pad+1):
                    xx = min(max(x + dx, 0), w - 1)
                    yy = min(max(y + dy, 0), h - 1)
                    weight = kernel[dy + pad][dx + pad]
                    r, g, b = pixels[xx, yy]
                    r_sum += r * weight
                    g_sum += g * weight
                    b_sum += b * weight
            # 寫入輸出（四捨五入並限制 0~255）
            out_pix[x, y] = (
                min(255, max(0, int(r_sum + 0.5))),
                min(255, max(0, int(g_sum + 0.5))),
                min(255, max(0, int(b_sum + 0.5))),
            )

    return out

def main():
    if len(sys.argv) < 3:
        print(f"用法：python {sys.argv[0]} input.jpg output.jpg [--ksize N] [--sigma S]")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]

    # 預設參數
    ksize = 5
    sigma = 1.0

    # 簡單參數解析
    args = sys.argv[3:]
    if '--ksize' in args:
        idx = args.index('--ksize')
        ksize = int(args[idx+1])
    if '--sigma' in args:
        idx = args.index('--sigma')
        sigma = float(args[idx+1])

    if ksize % 2 == 0:
        print("錯誤：ksize 必須為奇數")
        sys.exit(1)

    # 1. 讀影像並轉 RGB
    img = Image.open(in_path).convert('RGB')

    # 2. 生成 Gaussian kernel
    kernel = make_gaussian_kernel(ksize, sigma)

    # 3. 應用高斯模糊
    blurred = apply_gaussian_blur(img, kernel)

    # 4. 存檔
    blurred.save(out_path, format='JPEG')
    print(f"已完成高斯模糊，結果存為：{out_path}")

if __name__ == "__main__":
    main()
