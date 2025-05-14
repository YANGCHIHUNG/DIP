# Panorama Stitching Project

此專案示範如何使用 Python 與 OpenCV 實作全景拼接（Panorama Stitching）。

## 功能
- 載入多張有重疊區域的影像
- 偵測 SIFT 特徵並進行匹配
- 估算單應性矩陣（Homography）
- 投影、融合並輸出最終全景圖

## 安裝
```bash
pip install -r requirements.txt

panorama_project/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── input/
│   └── output/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── loader.py
│   ├── feature.py
│   ├── matcher.py
│   ├── estimator.py
│   ├── blender.py
│   └── stitcher.py
├── tests/
│   ├── test_loader.py
│   ├── test_feature.py
│   ├── test_matcher.py
│   └── test_stitcher.py
└── utils/
    ├── calibrator.py
    └── cropper.py


python -m src.main --config configs/default.yaml --input data/input --output data/output/panorama

流程與步驟

1. Load
    - 掃描指定資料夾、載入影像，回傳`List[ndarray]`。
2. Feature（特徵偵測）
    - 轉為灰階影像
    - 建立 SIFT 偵測器
    - 偵測並回傳偵測點與計算描述子（強制轉為 list，避免 OpenCV 回傳 tuple）
3. Matcher（特徵配對） 
    - 暴力匹配 (Brute-Force) 或 FLANN-based
    - 輸入兩張影像的 descriptors，輸出初步的 matches。
    - 可設定 ratio test（Lowe’s ratio）濾除不可靠的匹配對。

4. Estimator 單應矩陣估計（RANSAC）
    - 以 RANSAC 方法估計兩張影像間的單應矩陣 H：
        1. 隨機抽樣 4 對匹配點
        2. 計算單應矩陣
        3. 計算所有匹配點的重投影誤差，若誤差 < ransac_threshold 則為 inlier 
        4. 重複迭代，選擇 inlier 數最多的模型
    - 回傳最終的 H 與 inlier mask。
5. 影像投影與拼接
6. 影像融合 (Blending)
7. 裁切 (Cropping)
8. 輸出結果

