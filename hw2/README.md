# Panorama Stitching Project

此專案示範如何使用 Python 與 OpenCV 實作全景拼接（Panorama Stitching）。

## 功能
- 載入多張有重疊區域的影像
- 偵測 SIFT 特徵並進行匹配
- 估算單應性矩陣（Homography）
- 投影、融合並輸出最終全景圖

panorama_stitcher/
├── data/
│   ├── input/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── img3.jpg
│   └── output/
│       └── panorama.jpg
│
├── config/
│   └── settings.yaml         # 拼接參數（feature detector, matcher, RANSAC 閾值…）
│
├── src/
│   ├── __init__.py
│   ├── loader.py             # 影像讀取與預處理
│   ├── feature.py            # 特徵偵測與描述子封裝（ORB/SIFT/AZKZE）
│   ├── matcher.py            # 特徵匹配策略（BF/FLANN + 篩選）
│   ├── transformer.py        # 仿射/單映射矩陣估算
│   ├── blender.py            # 多頻帶融合或羽化實作
│   └── stitcher.py           # 主拼接流程：串接 Loader→Feature→Matcher→Transformer→Blender
│
├── notebooks/
│   └── demo.ipynb            # Jupyter 示範，方便調整參數、直觀顯示中間結果
│
├── tests/
│   └── test_stitcher.py      # 單元測試（讀圖、特徵數量、拼接結果大小等）
│
├── logs/
│   └── stitcher.log          # 執行時記錄（可用 logging 模組輸出）
│
├── requirements.txt          # 專案相依套件清單
├── setup.py                  # 如需打包發佈可用
├── README.md                 # 專案說明與使用教學
└── LICENSE

## 安裝
```bash
pip install -r requirements.txt


python stitcher.py \
  --config settings.yaml \
  --input input \
  --output output/panorama.jpg