# DIP 影像處理作業 1

本專案實作兩種 JPEG 影像合併演算法，分別為「平均值合併」與「中位數合佈」，最終輸出 PPM 格式影像。

## 目錄結構
```
hw1/
├─ include/
│  ├─ jpeg_reader.h
│  └─ save_ppm.h
├─ src/
│  ├─ avg.cpp        // 平均值合併
│  ├─ median.cpp     // 中位數合併
│  ├─ jpeg_reader.cpp// JPEG 讀取實作
│  └─ save_ppm.cpp   // PPM 輸出實作
├─ build_avg.sh      // 編譯 avg
└─ build_median.sh   // 編譯 median
```  

## 相依性

- C++11  
- libjpeg  
- POSIX (mkdir)

## 快速開始

```bash
# 進入專案資料夾
cd hw1

# 建置平均值合併執行檔\ nsh build_avg.sh

# 建置中位數合併執行檔\ nsh build_median.sh
```

執行程式，傳入多個 JPEG 檔案路徑：

```bash
./avg image1.jpg image2.jpg image3.jpg
```  

結果會輸出至 `output/` 資料夾，檔名格式：

- **平均值：** `output/avg_result_<N>.ppm`  
- **中位數：** `output/median_result_<N>.ppm`  

其中 `<N>` 為實際處理的圖片數量。

## 主要程式檔案

- **avg.cpp**  
  平均值合併：讀取所有 JPEG，累加各像素值後除以張數，四捨五入後限縮至 `[0,255]`，輸出 PPM。

- **median.cpp**  
  中位數合併：為每個像素位置蒐集所有影像值，排序後取中位（偶數時取右側），輸出 PPM。

- **jpeg_reader.cpp / jpeg_reader.h**  
  使用 libjpeg 解碼 JPEG，回傳原始的 RGB(A) 資料。

- **save_ppm.cpp / save_ppm.h**  
  將原始緩衝輸出成 PPM (P3 ASCII) 格式檔案。

## 範例

### 平均值合併

```bash
./avg sample1.jpg sample2.jpg sample3.jpg
# 完成！結果存入 output/avg_result_3.ppm
```

### 中位數合併

```bash
./median sample1.jpg sample2.jpg sample3.jpg
# 完成！結果存入 output/median_result_3.ppm
```

