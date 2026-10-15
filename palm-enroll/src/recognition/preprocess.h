#pragma once
// Preprocessing identical to MatConvNet's vl_imreadjpeg(..., 'Resize', [H W]),
// which EE-PRnet was trained with. See Transform Model/python/vl_imread.py for the derivation.
#include <opencv2/core.hpp>

namespace palm {

// rgb32f: CV_32FC3 (any channel count works), values 0..255.
// Returns CV_32FC3 of size `out`, float, not rounded.
cv::Mat vlResize(const cv::Mat& rgb32f, cv::Size out);

// BGR uint8 (cv::imread) -> RGB CV_32FC3 0..255.
cv::Mat toRgbFloat(const cv::Mat& bgr8);

// HWC float -> CHW float, dst must hold rows * cols * channels values.
void packChw(const cv::Mat& hwc32f, float* dst);

}  // namespace palm
