'''
python adaptive_median_filter.py /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/noisy_01.jpg /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/adaptive_denoisy_01.jpg -m 3
'''
import cv2
import numpy as np

def adaptive_median_filter(img: np.ndarray, max_ksize: int = 7) -> np.ndarray:
    """
    自適應中值濾波 (Adaptive Median Filter) 核心，僅支援單通道灰階。

    參數：
    - img: 灰階影像 (H, W)，dtype 任意整數型別
    - max_ksize: 最大視窗大小 (必須為奇數，>=3)

    回傳：
    - filtered: 濾波後灰階影像，同 dtype
    """
    assert img.ndim == 2, "adaptive_median_filter 只支援單通道"
    assert max_ksize % 2 == 1 and max_ksize >= 3, "max_ksize 必須為大於等於 3 的奇數"

    H, W = img.shape
    Smax = max_ksize
    filtered = np.zeros_like(img)
    pad = Smax // 2
    padded = np.pad(img, pad, mode='edge')

    for y in range(H):
        for x in range(W):
            S = 3
            z0 = int(padded[y + pad, x + pad])
            while True:
                off = S // 2
                window = padded[
                    y + pad - off : y + pad + off + 1,
                    x + pad - off : x + pad + off + 1
                ].ravel()
                Zmin = int(window.min())
                Zmax = int(window.max())
                Zmed = int(np.median(window))
                A1 = Zmed - Zmin
                A2 = Zmed - Zmax
                # Stage A
                if A1 > 0 and A2 < 0:
                    # Stage B
                    B1 = z0 - Zmin
                    B2 = z0 - Zmax
                    filtered[y, x] = z0 if (B1 > 0 and B2 < 0) else Zmed
                    break
                S += 2
                if S > Smax:
                    filtered[y, x] = Zmed
                    break
    return filtered


def adaptive_median_filter_color(input_path: str, output_path: str, max_ksize: int = 7) -> None:
    """
    讀取彩色影像，拆通道獨立做自適應中值濾波，重建彩色輸出。

    參數：
    - input_path: 輸入 BGR 影像路徑
    - output_path: 濾波後 BGR 影像儲存路徑
    - max_ksize: 最大視窗大小 (奇數，>=3)
    """
    # 讀彩色，強制 BGR
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"找不到影像：{input_path}")

    # 拆成三個通道
    b, g, r = cv2.split(img_bgr)

    # 對每個通道做自適應中值濾波
    b_d = adaptive_median_filter(b, max_ksize).astype(b.dtype)
    g_d = adaptive_median_filter(g, max_ksize).astype(g.dtype)
    r_d = adaptive_median_filter(r, max_ksize).astype(r.dtype)

    # 合併回 BGR
    denoised = cv2.merge([b_d, g_d, r_d])

    # 存檔
    cv2.imwrite(output_path, denoised)
    print(f"已將拆通道自適應中值濾波後彩色影像儲存至：{output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Median Filter for Color Images (Channel-wise)")
    parser.add_argument("input", help="輸入影像檔路徑 (BGR)")
    parser.add_argument("output", help="輸出影像檔路徑")
    parser.add_argument(
        "-m", "--max", type=int, default=7,
        help="最大視窗大小 (奇數，>=3)"
    )
    args = parser.parse_args()

    if args.max < 3 or args.max % 2 == 0:
        parser.error("max 必須為大於等於 3 的奇數")

    adaptive_median_filter_color(args.input, args.output, args.max)
