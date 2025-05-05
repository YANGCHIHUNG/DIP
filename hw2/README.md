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