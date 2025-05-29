import numpy as np
import matplotlib.pyplot as plt

def make_triangle_se(side):
    """Generate an upright right equilateral triangle structuring element."""
    h = int(round(np.sqrt(3) / 2 * side))
    se = np.zeros((h, side), dtype=np.uint8)
    for r in range(h):
        span = int((side / 2) * (1 - (r / h)))
        left = side // 2 - span
        right = side // 2 + span
        se[r, left:right+1] = 1
    return se

def dilation_binary(img, se, anchor):
    H, W = img.shape
    h, w = se.shape
    ax, ay = anchor
    out_H, out_W = H + h - 1, W + w - 1
    out = np.zeros((out_H, out_W), dtype=np.uint8)

    for i in range(out_H):
        for j in range(out_W):
            hit = False
            for u in range(h):
                for v in range(w):
                    if se[u, v] == 0:
                        continue
                    x = i - u + ax
                    y = j - v + ay
                    if 0 <= x < H and 0 <= y < W and img[x, y] == 1:
                        hit = True
                        break
                if hit:
                    break
            out[i, j] = 1 if hit else 0
    return out

# 1. 建立 200×200 的二值方形圖像
img = np.ones((200, 200), dtype=np.uint8)

# 2. 建立 30px 正三角形 SE
side = 30
se = make_triangle_se(side)

# 3. 計算 Anchor (三角形重心)
h = se.shape[0]
anchor = (int(round(h / 3)), side // 2)

# 4. 執行膨脹
dilated = dilation_binary(img, se, anchor)

# 5. 繪製膨脹結果
plt.figure()
plt.imshow(dilated)
plt.axis('off')
plt.title('Dilation Result')
plt.show()