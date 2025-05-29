import cv2
import argparse

def convert_to_grayscale(input_path: str, output_path: str) -> None:
    """
    讀取彩色影像，轉成灰階後儲存。

    參數：
    - input_path: 輸入彩色影像檔路徑（BGR）
    - output_path: 輸出灰階影像檔路徑
    """
    # 以 BGR 模式讀取
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"找不到影像：{input_path}")

    # 轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 儲存灰階結果
    cv2.imwrite(output_path, gray)
    print(f"已儲存灰階影像至：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="彩色影像轉灰階範例")
    parser.add_argument("input", help="輸入影像檔路徑")
    parser.add_argument("output", help="輸出灰階影像檔路徑")
    args = parser.parse_args()

    convert_to_grayscale(args.input, args.output)