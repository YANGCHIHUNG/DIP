#include "jpeg_reader.h"
#include <cstdio>
#include <iostream>

// 定義在 header 中宣告的函式
unsigned char* read_JPEG_file(const char* filename,
                              int &width,
                              int &height,
                              int &channels)
{
    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        std::cerr << "Error: 無法開啟檔案 " << filename << std::endl;
        return nullptr;
    }

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    width    = cinfo.output_width;
    height   = cinfo.output_height;
    channels = cinfo.output_components;
    int row_stride = width * channels;

    unsigned char* buffer =
        new unsigned char[width * height * channels];

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* rowptr =
            buffer + cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return buffer;
}
