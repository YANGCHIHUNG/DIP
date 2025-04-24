#include <iostream>
#include <vector>
#include <string>         // for std::to_string
#include "jpeg_reader.h"
#include "save_ppm.h"
#include <sys/stat.h>
#include <algorithm>      // for std::sort

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_count>\n";
        return 1;
    }
    int totalImages = 0;
    try {
        totalImages = std::stoi(argv[1]);
    } catch (...) {
        std::cerr << "請提供合法的整數作為圖片數量。\n";
        return 1;
    }
    if (totalImages <= 0) {
        std::cerr << "圖片數量必須大於 0。\n";
        return 1;
    }

    int width = 0, height = 0, channels = 0;
    int validCount = 0;
    std::vector<std::vector<int>> pixelValues;  // 每個像素位置的所有影像值
    const std::string imgDir = "img";

    for (int i = 1; i <= totalImages; ++i) {
        std::string path = imgDir + "/img" + std::to_string(i) + ".jpg";
        int w, h, c;
        unsigned char* buf = read_JPEG_file(path.c_str(), w, h, c);
        if (!buf) {
            std::cerr << "讀取失敗，跳過 " << path << "\n";
            continue;
        }

        if (validCount == 0) {
            width = w; height = h; channels = c;
            int pixelCount = width * height * channels;
            pixelValues.assign(pixelCount, std::vector<int>());
        } else if (w != width || h != height || c != channels) {
            std::cerr << "警告：尺寸不符，跳過 " << path << "\n";
            delete[] buf;
            continue;
        }

        int pixelCount = width * height * channels;
        for (int j = 0; j < pixelCount; ++j) {
            pixelValues[j].push_back(buf[j]);
        }
        delete[] buf;
        ++validCount;
    }

    if (validCount == 0) {
        std::cerr << "沒有可用的圖片進行中位數運算。\n";
        return 1;
    }

    // 計算每個像素的中位數
    int pixelCount = width * height * channels;
    std::vector<unsigned char> medianBuf(pixelCount);
    for (int j = 0; j < pixelCount; ++j) {
        auto &vals = pixelValues[j];
        std::sort(vals.begin(), vals.end());
        int mid = validCount / 2;
        int med = vals[mid];  // 偶數時選中間偏右的元素
        medianBuf[j] = static_cast<unsigned char>(std::min(med, 255));
    }

    // 確保 output 資料夾存在
    mkdir("output", 0755);

    // 建立輸出檔名：avg_result_<validCount>.ppm
    std::string outPath = "output/median_result_" + std::to_string(validCount) + ".ppm";

    if (!save_PPM(outPath.c_str(), medianBuf.data(), width, height, channels)) {
        std::cerr << "儲存結果失敗：" << outPath << "\n";
        return 1;
    }

    std::cout << "完成！結果存入 " << outPath << std::endl;
    return 0;
}
