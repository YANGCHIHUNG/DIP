#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from PIL import Image

def convert_ppm_to_jpg(input_dir: str, output_dir: str, quality: int = 85):
    """
    將 input_dir 中所有 .ppm 檔轉成 .jpg，並存到 output_dir。

    參數:
    - input_dir: ppm 檔所在資料夾 (e.g. "img")
    - output_dir: jpg 要存放的資料夾 (e.g. "static")
    - quality: 輸出 jpg 的品質 (1–95)
    """
    # 若輸出資料夾不存在就建立
    os.makedirs(output_dir, exist_ok=True)

    # 遍歷所有檔案
    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.ppm'):
            src_path = os.path.join(input_dir, fname)
            dst_name = os.path.splitext(fname)[0] + '.jpg'
            dst_path = os.path.join(output_dir, dst_name)

            # 開啟 PPM，轉為 RGB，再存為 JPEG
            with Image.open(src_path) as im:
                rgb = im.convert('RGB')
                rgb.save(dst_path, 'JPEG', quality=quality)

            print(f'已轉換：{src_path} → {dst_path}')

if __name__ == '__main__':
    # 定義輸入/輸出資料夾
    input_folder = 'output'
    output_folder = 'static'

    convert_ppm_to_jpg(input_folder, output_folder)
