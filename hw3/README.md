這份作業的核心目標是「結合一階與二階微分的邊緣資訊，並搭配去雜訊與權重融合，最後把加強後的邊緣效果回補到原始影像中」，以達到細節增強又不過度放大雜訊的影像強化效果。具體流程可分為以下幾個階段：

1. **二階微分邊緣偵測（拉普拉斯）**

   * 用 Laplacian（或 LoG）算子計算二階導數，強調影像中灰階急劇變化處（尖銳邊緣）。
   * 取得的 `edge2` 圖只保留細節與邊緣輪廓。

2. **二階邊緣加回原圖（銳化）**

   * 把 `edge2` 與原圖相加：

     $$
       \text{sharp2} = \text{原圖} + \alpha \times \text{edge2}
     $$
   * 得到初步的銳化影像 `sharp2`，細節更明顯，但也可能有雜訊放大。

3. **一階微分邊緣偵測（梯度）**

   * 用 Sobel/Prewitt 算子計算一階導數，取得連續性更好的邊緣圖 `edge1`。
   * 這張圖包含較多雜訊偵測結果，需要後續去除。

4. **對一階邊緣圖做去雜訊（模糊）**

   * 用高斯模糊（或其他濾波）把 `edge1` 平滑成 `smooth1`，消除假邊緣與雜訊。
   * 這裡 `smooth1` 將作為「權重遮罩」。

5. **將去噪後邊緣圖正規化成 0～1 的權重圖**

   * 把 `smooth1` 最小–最大正規化：

     $$
       w(x,y) = \frac{\smooth1(x,y) - \min}{\max - \min}
     $$
   * 轉成「每一像素的加強係數」。

6. **權重圖乘上銳化影像，再加回原圖 → 最終影像 A**

   * 計算：

     $$
       \text{fusionA} = w \times \text{sharp2},\quad
       \text{resultA} = \text{原圖} + \beta \times \text{fusionA}
     $$
   * 讓一階邊緣強度高的區域進一步放大二階銳化效果，同時保留原圖基底。

7. **權重圖乘上二階邊緣圖，再加回原圖 → 最終影像 B**

   * 計算：

     $$
       \text{fusionB} = w \times \text{edge2},\quad
       \text{resultB} = \text{原圖} + \gamma \times \text{fusionB}
     $$
   * 另一種變體：依照一階平滑後的權重來選擇性放大純二階邊緣資訊。

---

### 作業重點

* **理解微分階數差異**：

  * 一階微分（梯度）：邊緣連續、方向性明顯，雜訊敏感度中等。
  * 二階微分（Laplacian）：精準定位「尖銳」變化，對雜訊特別敏感。
* **去雜訊—正規化—加權融合**：

  * 先將一階邊緣去雜訊、轉成權重遮罩，避免在雜訊區域過度放大。
  * 再用這張權重圖去選擇性強化二階或一階銳化結果。
* **實作細節**：

  * 核心算子：Sobel／Prewitt、Gaussian Blur、Laplacian。
  * 正規化方式：最小–最大線性映射到 \[0,1]。
  * 加回原圖時可再乘以可調係數 α, β, γ，控制最終加強強度。

完成後，應提交：

1. 實作程式（或 Notebook）
2. 中間結果影像（edge1、edge2、smooth1、fusionA、resultA、fusionB、resultB）
3. 簡短說明文件（各步驟原理、參數選擇依據、視覺效果比較）


python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i /Users/young/Documents/nchu-2025-spring/DIP/hw3/data/input/img.png \
  --outscale 4 \
  --face_enhance


python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i /Users/young/Documents/nchu-2024-fall/LA/HW4/SaveInsta.to_476488059_18357903655135549_9131945504073213709_n.jpg \
  --outscale 4 \
  --tile 200 \
  --tile_pad 10


python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  -i /Users/young/Documents/nchu-2025-spring/DIP/hw3/input/img.png \
  --outscale 4 \
  --tile 200 \
  --tile_pad 10



圖像A: 二階微分邊緣圖 先加上 原圖           再乘上 一階微分正規化圖
圖像B: 二階微分邊緣圖 先乘上 一階微分正規化圖 再加上 原圖
圖像C: Super resolution

Adaptive Median Filter vs Median Filter

椒鹽雜訊
P0 = P255 = 1/10, 使用3*3
P0 = P255 = 1/4, 使用7*7

AMF 和 MF 都是使用 3*3 或 7*7 嗎？

