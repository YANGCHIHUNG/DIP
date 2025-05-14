#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sobel.py

python sobel.py input_image output.jpg [quality]
python sobel.py /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/blur_img.png /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/output/sobel_blur_img.png

功能：
1. 用 Pillow 讀入任意影像檔 (jpg/png/pgm...)
2. 純 Python 實作 Sobel 邊緣偵測
3. 輸出為 JPEG（quality 可自訂）
"""

import sys
import math
from PIL import Image

# Sobel 核定義
Gx = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]
Gy = [
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
]

def sobel_edge_detection(pixels, w, h, maxval=255):
    """
    image_pxls: 2D list of gray values (0~maxval)
    回傳 2D list of Sobel 邊緣強度 (0~maxval)
    """
    output = [[0]*w for _ in range(h)]
    for y in range(1, h-1):
        for x in range(1, w-1):
            gx = gy = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    v = pixels[y+dy][x+dx]
                    gx += Gx[dy+1][dx+1] * v
                    gy += Gy[dy+1][dx+1] * v
            mag = math.hypot(gx, gy)  # sqrt(gx^2+gy^2)
            output[y][x] = int(min(maxval, max(0, mag)))
    return output

def image_to_pixels(img):
    """
    把 Pillow 灰階 Image 轉成 2D list
    """
    w, h = img.size
    pix = img.load()
    return [[pix[x, y] for x in range(w)] for y in range(h)]

def pixels_to_image(pixels, w, h):
    """
    把 2D list 轉回 Pillow 灰階 Image
    """
    img = Image.new('L', (w, h))
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), pixels[y][x])
    return img

def main():
    if len(sys.argv) < 3:
        print(f'用法：python {sys.argv[0]} input_image output.jpg [quality]')
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2]
    quality  = int(sys.argv[3]) if len(sys.argv) >= 4 else 95

    # 1. 讀取並轉灰階
    img = Image.open(in_path).convert('L')
    w, h = img.size

    # 2. 灰階影像→2D list
    pixels = image_to_pixels(img)

    # 3. Sobel 偵測
    edges = sobel_edge_detection(pixels, w, h, maxval=255)

    # 4. 2D list→Pillow Image
    edge_img = pixels_to_image(edges, w, h)

    # 5. 儲存為 JPEG
    edge_img.save(out_path, format='JPEG', quality=quality)
    print(f'已完成 Sobel 偵測，結果存為：{out_path} (quality={quality})')

if __name__ == '__main__':
    main()
