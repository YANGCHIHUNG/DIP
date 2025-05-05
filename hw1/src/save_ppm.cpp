#include "save_ppm.h"
#include <fstream>
#include <iostream>
#include <algorithm> // for std::min

bool save_PPM(const char* filename,
              const unsigned char* data,
              int width,
              int height,
              int channels)
{
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: 無法寫入檔案 " << filename << std::endl;
        return false;
    }

    // 只支援灰階 (channels=1) 或 RGB (channels=3)
    if (channels == 1) {
        ofs << "P2\n";
    } else {
        ofs << "P3\n";
    }
    ofs << width << " " << height << "\n255\n";

    int pixelCount = width * height * channels;
    for (int i = 0; i < pixelCount; ++i) {
        int v = static_cast<int>(data[i]);
        v = std::min(v, 255);
        ofs << v;
        if ((i + 1) % channels != 0) {
            ofs << " ";
        } else {
            ofs << "\n";
        }
    }

    ofs.close();
    return true;
}
