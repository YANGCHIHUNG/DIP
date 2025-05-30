'''
python adaptive_median_filter.py /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/noisy_low.png /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/AMF_low.png -m 15
python adaptive_median_filter.py /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/noisy_high.png /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/AMF_high.png -m 15
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import sys

def adaptive_median_filter(img: np.ndarray, s_max: int) -> np.ndarray:
    """
    自適應中值濾波器 (輸入單通道灰階影像)
    img    : np.ndarray, 灰階影像
    s_max  : int, 最大視窗邊長 (必須為奇數)
    回傳   : np.ndarray, 除躁後的灰階影像
    """

    from collections import Counter
    counts = Counter()

    # Padding：以最大視窗半徑作零填充
    pad = s_max // 2
    padded = np.pad(img, pad_width=pad, mode='reflect')
    output = img.copy()
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            window_size = 3  # 從 3×3 開始
            while True:
                half = window_size // 2
                # 取出當前視窗
                local = padded[i : i + window_size, j : j + window_size]
                z_min = local.min()
                z_med = np.median(local)
                z_max = local.max()
                z_xy  = padded[i + half, j + half]

                # 新增：計算非噪點個數
                vals = local.flatten()
                good = np.count_nonzero((vals != 0) & (vals != 255))                

                # Step A: 檢查中值是否在最小值與最大值之間and good >= 21
                if z_min < z_med < z_max :
                    # Step B: 檢查當前像素是否等於中值
                    output[i, j] = z_med if z_xy != z_med else z_xy
                    counts[window_size] += 1
                    break
                else:
                    window_size += 2  # 每次擴增視窗邊長 2
                    if window_size > s_max:
                        # 超過最大視窗時，直接以中值替代
                        output[i, j] = z_med
                        counts[window_size] += 1
                        break    

    print("視窗大小使用次數：")
    for size in sorted(counts):
        print(f"{size}x{size}: {counts[size]} 次")         

    return output

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Median Filter (灰階版)\n"
                    "範例：python adaptive_median_filter.py input.png output.png -m 31"
    )
    parser.add_argument("input_path", help="輸入影像路徑 (灰階)")
    parser.add_argument("output_path", help="輸出影像路徑")
    parser.add_argument(
        "-m", "--max_window", type=int, default=31,
        help="最大視窗邊長 (預設 31，必須為奇數)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 以灰階讀取
    img = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"無法讀取影像：{args.input_path}", file=sys.stderr)
        sys.exit(1)

    # 檢查 max_window 是否為奇數
    if args.max_window % 2 == 0:
        print("錯誤：-m/--max_window 必須為奇數", file=sys.stderr)
        sys.exit(1)

    denoised = adaptive_median_filter(img, args.max_window)

    # 儲存結果（灰階）
    success = cv2.imwrite(args.output_path, denoised)
    if not success:
        print(f"儲存影像失敗：{args.output_path}", file=sys.stderr)
        sys.exit(1)

    print(f"除躁完成，已儲存至：{args.output_path}")

if __name__ == "__main__":
    main()