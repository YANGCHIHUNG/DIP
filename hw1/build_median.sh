#!/usr/bin/env bash
set -e

clang++ -std=c++11 \
  src/median.cpp src/jpeg_reader.cpp src/save_ppm.cpp \
  -Iinclude \
  -I/opt/homebrew/opt/jpeg/include \
  -L/opt/homebrew/opt/jpeg/lib -ljpeg \
  -o median

echo "Build completed: ./median"
