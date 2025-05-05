#ifndef JPEG_READER_H
#define JPEG_READER_H

#include <cstddef>   // for size_t
#include <cstdio>    // for FILE

#include <jpeglib.h>

// 讀取 JPEG 檔案，返回原始像素資料，並通過參考參數設定寬、高、通道數
// 呼叫端記得最後用 delete[] 釋放回傳的緩衝區
unsigned char* read_JPEG_file(const char* filename,
                              int &width,
                              int &height,
                              int &channels);

#endif // JPEG_READER_H
