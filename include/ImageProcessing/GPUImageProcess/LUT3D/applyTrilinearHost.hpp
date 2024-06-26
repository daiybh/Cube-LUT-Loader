#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>

namespace GpuTrilinear
{
cv::Mat applyTrilinearGpu(cv::Mat input, const Table3D &lut, const float opacity, const int threads);
}
