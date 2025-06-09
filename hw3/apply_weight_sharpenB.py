#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_weight_sharpenB.py

功能：
  根據一階邊緣權重圖，在邊緣強的地方將二階邊緣圖(edge2)加回原始彩色影像，
  其餘區域維持原始影像不變，產生 resultB。

使用：
  pip install pillow
  python apply_weight_sharpenB.py \
      --original orig.png \
      --edge2 edge2.png \
      --weight weight.png \
      --output resultB.png \
      [--gamma 1.0]

    python apply_weight_sharpenB.py \
        --original /Users/young/Documents/nchu-2025-spring/DIP/hw3/input/img.png \
        --edge2 /Users/young/Documents/nchu-2025-spring/DIP/hw3/output/edge2.png \
        --weight /Users/young/Documents/nchu-2025-spring/DIP/hw3/output/weight.png \
        --output /Users/young/Documents/nchu-2025-spring/DIP/hw3/output/resultB.png

參數：
  --original  原始彩色影像路徑
  --edge2     二階邊緣圖 (灰階)
  --weight    一階邊緣權重圖 (灰階, 0~255 表示 0.0~1.0)
  --output    最終輸出影像路徑
  --gamma     邊緣加回強度係數 (預設 1.0)
"""
import sys
import argparse
from PIL import Image

print("TEst")

def parse_args():
    parser = argparse.ArgumentParser(description="融合一階權重與二階邊緣圖 (resultB)")
    parser.add_argument('--original', required=True, help='原始彩色影像')
    parser.add_argument('--edge2', required=True, help='二階邊緣圖 (灰階)')
    parser.add_argument('--weight', required=True, help='一階邊緣權重圖 (灰階)')
    parser.add_argument('--output', required=True, help='輸出結果影像')
    parser.add_argument('--gamma', type=float, default=1.0, help='邊緣加回強度 (預設 1.0)')
    return parser.parse_args()

def load_images(orig_path, edge2_path, weight_path):
    orig = Image.open(orig_path).convert('RGB')
    edge2 = Image.open(edge2_path).convert('L')
    weight = Image.open(weight_path).convert('L')
    if orig.size != edge2.size or orig.size != weight.size:
        raise ValueError('三張影像尺寸必須相同')
    return orig, edge2, weight

def apply_resultB(orig, edge2, weight, gamma):
    w, h = orig.size
    orig_pix = orig.load()
    edge_pix = edge2.load()
    weight_pix = weight.load()

    result = Image.new('RGB', (w, h))
    res_pix = result.load()

    for y in range(h):
        for x in range(w):
            w_val = weight_pix[x, y] / 255.0
            e_val = edge_pix[x, y]  # 0-255 edge intensity
            delta = int(gamma * w_val * e_val)
            r0, g0, b0 = orig_pix[x, y]
            # 加回並 clamp
            r = min(255, max(0, r0 + delta))
            g = min(255, max(0, g0 + delta))
            b = min(255, max(0, b0 + delta))
            res_pix[x, y] = (r, g, b)

    return result

def main():
    print("Starting...")
    args = parse_args()
    orig, edge2, weight = load_images(args.original, args.edge2, args.weight)
    result = apply_resultB(orig, edge2, weight, args.gamma)
    result.save(args.output, format='PNG', quality=95)
    print(f"已完成 resultB 融合，輸出：{args.output} (gamma={args.gamma})")

if __name__ == '__main__':

    print("Starting...")
    main()

