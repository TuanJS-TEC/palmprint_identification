#include "preprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace palm {
namespace {

struct Filter1D {
  std::vector<std::vector<std::pair<int, float>>> taps;

  Filter1D(int outSize, int inSize, int cropSize, int cropOffset) : taps(outSize) {
    const float alpha = static_cast<float>(cropSize) / outSize;
    const float beta = 0.5f * (alpha - 1) + cropOffset;
    float support = 2.f;  // bilinear
    int size = 2;
    if (alpha > 1) {
      support *= alpha;
      size = static_cast<int>(std::ceil(support));
    }
    std::vector<float> w(inSize);
    for (int v = 0; v < outSize; ++v) {
      const float u = alpha * v + beta;
      const int start = static_cast<int>(std::ceil(u - support / 2));
      float mass = 0.f;
      std::fill(w.begin(), w.end(), 0.f);
      int lo = inSize, hi = -1;
      for (int r = 0; r < size; ++r) {
        const int k = start + r;
        float delta = u - k;
        if (alpha > 1) delta /= alpha;
        const float h = std::max(0.f, 1.f - std::fabs(delta));
        const int q = std::clamp(k, 0, inSize - 1);
        w[q] += h;
        mass += h;
        if (h != 0.f) lo = std::min(lo, q), hi = std::max(hi, q);
      }
      for (int q = lo; q <= hi; ++q)
        if (w[q] != 0.f) taps[v].emplace_back(q, w[q] / mass);
    }
  }
};

cv::Rect vlCropGeometry(cv::Size in, cv::Size out) {
  const double scale = std::min(static_cast<double>(in.width) / out.width,
                                static_cast<double>(in.height) / out.height);
  const int w = static_cast<int>(std::min(std::round(out.width * scale), static_cast<double>(in.width)));
  const int h = static_cast<int>(std::min(std::round(out.height * scale), static_cast<double>(in.height)));
  return {(in.width - w) / 2, (in.height - h) / 2, w, h};
}

}  // namespace

cv::Mat vlResize(const cv::Mat& src, cv::Size out) {
  if (src.empty() || src.depth() != CV_32F) throw std::invalid_argument("vlResize: expects a CV_32F image");
  const cv::Mat in = src.isContinuous() ? src : src.clone();
  const int cn = in.channels(), inW = in.cols, inH = in.rows;
  const cv::Rect crop = vlCropGeometry(in.size(), out);
  const Filter1D fv(out.height, inH, crop.height, crop.y);
  const Filter1D fh(out.width, inW, crop.width, crop.x);

  // vertical pass: out.height x inW
  cv::Mat tmp(out.height, inW, CV_32FC(cn), cv::Scalar::all(0));
  for (int v = 0; v < out.height; ++v) {
    float* dst = tmp.ptr<float>(v);
    for (const auto& [k, wt] : fv.taps[v]) {
      const float* row = in.ptr<float>(k);
      for (int i = 0; i < inW * cn; ++i) dst[i] += wt * row[i];
    }
  }
  // horizontal pass: out.height x out.width
  cv::Mat dst(out, CV_32FC(cn), cv::Scalar::all(0));
  for (int v = 0; v < out.height; ++v) {
    const float* row = tmp.ptr<float>(v);
    float* o = dst.ptr<float>(v);
    for (int u = 0; u < out.width; ++u)
      for (const auto& [k, wt] : fh.taps[u])
        for (int c = 0; c < cn; ++c) o[u * cn + c] += wt * row[k * cn + c];
  }
  return dst;
}

cv::Mat toRgbFloat(const cv::Mat& bgr8) {
  if (bgr8.empty() || bgr8.type() != CV_8UC3) throw std::invalid_argument("toRgbFloat: expects a CV_8UC3 BGR image");
  cv::Mat rgb, f;
  cv::cvtColor(bgr8, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(f, CV_32FC3);
  return f;
}

void packChw(const cv::Mat& hwc, float* dst) {
  const int cn = hwc.channels();
  const size_t plane = hwc.total();
  for (int y = 0; y < hwc.rows; ++y) {
    const float* p = hwc.ptr<float>(y);
    for (int x = 0; x < hwc.cols; ++x)
      for (int c = 0; c < cn; ++c) dst[c * plane + static_cast<size_t>(y) * hwc.cols + x] = p[x * cn + c];
  }
}

}  // namespace palm
