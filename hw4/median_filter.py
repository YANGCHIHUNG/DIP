'''
python median_filter.py /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/noisy_low.png /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/denoised_low.png -k 3
python median_filter.py /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/noisy_high.png /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/denoised_high.png -k 7

'''
import cv2
import numpy as np

def manual_median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    對輸入的影像陣列做中值濾波（純手動實作，不用 cv2.medianBlur）。
    
    參數：
    - img: 輸入影像，形狀為 (H, W) 或 (H, W, C)
    - ksize: 濾波視窗大小 (必須為正奇數，且 >= 3)
    
    回傳：
    - filtered: 同尺寸的中值濾波後影像
    """
    assert ksize % 2 == 1 and ksize >= 3, "ksize 必須為大於等於 3 的奇數"
    pad = ksize // 2
    
    # 若是灰階 (H, W)，加一個 channel 維度
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    H, W, C = img.shape
    
    # 邊界複製填充
    padded = np.pad(img, ((pad,pad), (pad,pad), (0,0)), mode='edge')
    out = np.zeros_like(img)
    
    # 逐像素、逐通道計算中值
    for y in range(H):
        for x in range(W):
            for c in range(C):
                window = padded[y:y+ksize, x:x+ksize, c].ravel()
                # 排序並取中值
                out[y, x, c] = np.median(window)
    
    # 如果原本是灰階，去掉最後的 channel 維度
    if out.shape[2] == 1:
        out = out[:, :, 0]
    return out

def median_denoise_cv2_io(input_path: str, output_path: str, ksize: int = 3) -> None:
    """
    讀取影像 (cv2)、做純手動中值濾波、儲存結果 (cv2)。
    """
    # 讀影像（BGR）
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"找不到影像：{input_path}")
    
    # 執行純手動中值濾波
    denoised = manual_median_filter(img, ksize=ksize).astype(np.uint8)
    
    # 存檔
    cv2.imwrite(output_path, denoised)
    print(f"已將去雜訊影像儲存至：{output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用純手動中值濾波 (cv2 只做 I/O)")
    parser.add_argument("input", help="輸入影像檔路徑")
    parser.add_argument("output", help="輸出影像檔路徑")
    parser.add_argument(
        "-k", "--ksize", type=int, default=3,
        help="中值濾波視窗大小 (奇數，預設 3)"
    )
    args = parser.parse_args()
    
    if args.ksize < 3 or args.ksize % 2 == 0:
        parser.error("ksize 必須為大於等於 3 的奇數")
    
    median_denoise_cv2_io(args.input, args.output, args.ksize)