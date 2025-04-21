#ifndef SAVE_PPM_H
#define SAVE_PPM_H

// 將 raw RGB(A) 資料輸出為 PPM (P3 ASCII) 格式
// filename: 輸出路徑
// data:    像素資料緩衝 (每像素 channels 個 byte)
// width:   圖寬
// height:  圖高
// channels:通道數 (建議 1 或 3)
// 呼叫端只需判斷回傳 true/false 即可
bool save_PPM(const char* filename,
              const unsigned char* data,
              int width,
              int height,
              int channels);

#endif // SAVE_PPM_H
