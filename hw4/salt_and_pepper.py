'''
python salt_and_pepper.py /Users/young/Documents/nchu-2025-spring/DIP/hw4/input/g_img.png /Users/young/Documents/nchu-2025-spring/DIP/hw4/output/noisy_25.jpg --amount 0.5
'''

import cv2
import numpy as np

def add_salt_and_pepper_noise(img: np.ndarray, amount: float) -> np.ndarray:
    """
    在影像上加入椒鹽雜訊。
    
    參數：
    - img: 輸入影像陣列，形狀可為 (H, W) 或 (H, W, C)
    - amount: 雜訊比例，代表多少比例的像素會被設為鹽或胡椒（0.0–1.0）
    
    回傳：
    - noisy: 加完雜訊後的影像，dtype 與原圖相同
    """
    assert 0.0 <= amount <= 1.0, "amount 必須介於 0.0 和 1.0 之間"
    
    noisy = img.copy()
    H, W = img.shape[:2]
    # 要加入雜訊的像素總數
    num_noisy = int(amount * H * W)
    # 一半做「鹽」（白），一半做「胡椒」（黑）
    num_salt = num_noisy // 2
    num_pepper = num_noisy - num_salt

    # 隨機選位置
    # salt
    coords = [
        np.random.randint(0, dim, size=num_salt)
        for dim in (H, W)
    ]
    noisy[coords[0], coords[1]] = 255

    # pepper
    coords = [
        np.random.randint(0, dim, size=num_pepper)
        for dim in (H, W)
    ]
    noisy[coords[0], coords[1]] = 0

    return noisy

def sp_noise_cv2_io(input_path: str, output_path: str, amount: float = 0.05) -> None:
    """
    讀取影像 (cv2)、加入椒鹽雜訊、儲存結果 (cv2)。
    
    參數：
    - input_path: 輸入影像檔路徑
    - output_path: 輸出影像檔路徑
    - amount: 雜訊比例，預設 0.05 (5%)
    """
    # 讀影像（保留原本通道與深度）
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"找不到影像：{input_path}")

    # 加入椒鹽雜訊
    noisy = add_salt_and_pepper_noise(img, amount).astype(img.dtype)

    # 存檔
    cv2.imwrite(output_path, noisy)
    print(f"已將含椒鹽雜訊影像儲存至：{output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="加入椒鹽雜訊範例程式 (cv2 只做 I/O)"
    )
    parser.add_argument("input", help="輸入影像檔路徑")
    parser.add_argument("output", help="輸出影像檔路徑")
    parser.add_argument(
        "-a", "--amount", type=float, default=0.05,
        help="雜訊比例 (0.0–1.0)，代表多少比例的像素會成為鹽或胡椒，預設 0.05"
    )
    args = parser.parse_args()

    if not (0.0 <= args.amount <= 1.0):
        parser.error("amount 必須介於 0.0 和 1.0 之間")

    sp_noise_cv2_io(args.input, args.output, args.amount)