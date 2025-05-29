# 影像處理作業 2

- **系級：** 資訊工程系一年級
- **姓名：** 楊啟弘
- **學號：** 7113056083

本作業是一個全景（Panorama）影像拼接工具，透過模組化設計將流程拆分為：讀圖、特徵偵測、特徵匹配、單應性估計、影像投影與融合等步驟。整體架構與運作流程如下：



## 一、專案目標、要求

輸入：多張重疊且視角略有差異的影像（需三張以上）
輸出：一張無縫接合的全景影像
要求：
- Affine transform 不可依靠套件

## 二、主要程式檔與模組職責

此專案將核心功能拆分為多個模組，各模組職責如下：

| 檔案                           | 功能說明                                               |
|--------------------------------------|------------------------------------------------------|
| `config/settings.yaml`               | 拼接參數設定（特徵偵測器、匹配器、RANSAC、混合方式等）         |
| `loader.py`                      | 影像讀取與前處理（批次載入、灰階轉換、調整大小）               |
| `feature.py`                     | 特徵偵測與描述子計算（ORB、SIFT、AKAZE 等）                  |
| `matcher.py`                     | 描述子匹配邏輯（BFMatcher、FlannBasedMatcher 與 Lowe’s ratio test） |
| `transformer.py`                 | 仿射矩陣估算（RANSAC）、影像仿射變形與矩陣合成                 |
| `blender.py`                     | 影像融合方法（羽化混合 feather、金字塔多頻帶 multiband）        |
| `stitcher.py`                    | 主拼接流程整合：讀設定、載入影像、估算變換、Warp、混合、輸出      |                                                                               |

## 三、工作流程

下面示意 Panorama 作業的執行流程，依序說明每個步驟的核心功能，並提供主要程式碼範例：

1. **讀入影像**

   ```python
   # main.py
   import os, cv2
   parser.add_argument('--input_dir', required=True)
   args = parser.parse_args()
   file_list = sorted(os.listdir(args.input_dir))
   images = [cv2.imread(os.path.join(args.input_dir, f)) for f in file_list]
   ```

   * 讀取指定資料夾所有影像，載入為 NumPy 陣列。

2. **特徵偵測與描述**

   ```python
   # feature.py
   import cv2

   def detect_and_compute(img, method='ORB'):
       if method == 'ORB':
           detector = cv2.ORB_create()
       else:
           detector = cv2.SIFT_create()
       keypoints, descriptors = detector.detectAndCompute(img, None)
       return keypoints, descriptors
   ```

   * 使用 ORB 或 SIFT 偵測關鍵點並計算描述子。

3. **特徵匹配**

   ```python
   # matcher.py
   import cv2

   def match_features(des1, des2):
       bf = cv2.BFMatcher()
       matches = bf.knnMatch(des1, des2, k=2)
       good = [m for m, n in matches if m.distance < 0.75 * n.distance]
       return good
   ```

   * 使用 BFMatcher + Lowe’s ratio test 篩選優質匹配。

4. **單應性估計**

   ```python
   # estimator.py
   import cv2
   import numpy as np

   def estimate_homography(kps1, kps2, matches, threshold=5.0):
       src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
       dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
       H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
       return H, mask
   ```

   * 以 RANSAC 剔除外點並估計 3×3 單應矩陣。

5. **影像投影**

   ```python
   # stitcher.py
   width = panorama.shape[1] + img.shape[1]
   height = max(panorama.shape[0], img.shape[0])
   warped = cv2.warpPerspective(img, H, (width, height))
   ```

   * 根據 H 對新影像做透視投影，擴展畫布以容納所有影像。

6. **影像融合**

   ```python
   # blender.py
   import numpy as np

   def warp_perspective_and_blend(base, warped):
       h_base, w_base = base.shape[:2]
       result = np.zeros_like(warped)
       result[:h_base, :w_base] = base

       mask = np.any(warped != 0, axis=2).astype(np.float32)[..., None]
       result = result * (1 - mask) + warped * mask
       return result.astype(base.dtype)
   ```

   * 建立遮罩並對重疊區做線性加權羽化融合。

7. **迭代累積**

   ```python
   # stitcher.py
   panorama = images[0]
   for img in images[1:]:
       kps1, des1 = detect_and_compute(panorama)
       kps2, des2 = detect_and_compute(img)
       matches = match_features(des1, des2)
       H, mask = estimate_homography(kps1, kps2, matches)
       warped = cv2.warpPerspective(img, H, (width, height))
       panorama = warp_perspective_and_blend(panorama, warped)
   ```

   * 以前一輪合併結果作為新基準，重複步驟 2–6，直到所有影像完成拼接。

## 四、成果範例展示

以下展示輸入影像與拼接後輸出結果示例，

| 影像 1 | 影像 2 | 影像 3 |
|:--------------:|:--------------:|:--------------:|
| ![範例輸入影像 1](../input/a.png) | ![範例輸入影像 2](../input/b.png) | ![範例輸入影像 3](../input/c.png) |

- **輸出全景圖**：`data/output/panorama_example.jpg`
  ![Example Output Panorama](../output/panorama.jpg)