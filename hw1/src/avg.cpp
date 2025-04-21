#include <iostream>
#include <vector>
#include <string>         // for std::to_string
#include "jpeg_reader.h"
#include "save_ppm.h"
#include <sys/stat.h>
#include <algorithm>      // for std::min

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "請提供至少一個 JPEG 檔案作為輸入。\n";
        return 1;
    }

    int totalImages = argc - 1;
    int width = 0, height = 0, channels = 0;
    std::vector<int> sumBuffer;
    int validCount = 0;

    for (int i = 1; i <= totalImages; ++i) {
        int w, h, c;
        unsigned char* buf = read_JPEG_file(argv[i], w, h, c);
        if (!buf) continue;

        if (validCount == 0) {
            width = w; height = h; channels = c;
            sumBuffer.assign(width * height * channels, 0);
        } else if (w != width || h != height || c != channels) {
            std::cerr << "警告：尺寸不符，跳過 " << argv[i] << "\n";
            delete[] buf;
            continue;
        }

        int pixelCount = width * height * channels;
        for (int j = 0; j < pixelCount; ++j) {
            sumBuffer[j] += buf[j];
        }
        delete[] buf;
        ++validCount;
    }

    if (validCount == 0) {
        std::cerr << "沒有可用的圖片進行平均運算。\n";
        return 1;
    }

    std::vector<unsigned char> avgBuf(width * height * channels);
    for (size_t i = 0; i < sumBuffer.size(); ++i) {
        int v = int(float(sumBuffer[i]) / validCount + 0.5f);
        avgBuf[i] = static_cast<unsigned char>(std::min(v, 255));
    }

    // 確保 output 資料夾存在
    mkdir("output", 0755);

    // 建立輸出檔名：avg_result_<validCount>.ppm
    std::string outPath = "output/avg_result_" + std::to_string(validCount) + ".ppm";

    if (!save_PPM(outPath.c_str(), avgBuf.data(), width, height, channels)) {
        std::cerr << "儲存結果失敗：" << outPath << "\n";
        return 1;
    }

    std::cout << "完成！結果存入 " << outPath << std::endl;
    return 0;
}
