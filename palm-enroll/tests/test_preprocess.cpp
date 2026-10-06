#include <cmath>

#include <opencv2/core.hpp>

#include "check.h"
#include "preprocess.h"

using namespace palm;

static double maxAbs(const cv::Mat& a, const cv::Mat& b) { return cv::norm(a, b, cv::NORM_INF); }

int main() {
  cv::RNG rng(1);

  // same size: the filter degenerates to the identity
  cv::Mat x(56, 56, CV_32FC3);
  rng.fill(x, cv::RNG::UNIFORM, 0, 255);
  CHECK(maxAbs(vlResize(x, {56, 56}), x) == 0);

  // constant image stays constant (weights sum to 1, borders replicated)
  const cv::Mat c(240, 400, CV_32FC3, cv::Scalar(77, 120, 3));
  const cv::Mat rc = vlResize(c, {56, 56});
  CHECK(rc.size() == cv::Size(56, 56));
  CHECK(maxAbs(rc, cv::Mat(56, 56, CV_32FC3, cv::Scalar(77, 120, 3))) < 1e-3);

  // vl_imreadjpeg crop: isotropic, centred, integer offset
  CHECK(vlCropGeometry({1920, 1080}, {224, 224}) == cv::Rect(420, 0, 1080, 1080));
  CHECK(vlCropGeometry({300, 301}, {56, 56}) == cv::Rect(0, 0, 300, 300));
  CHECK(vlCropGeometry({224, 224}, {56, 56}) == cv::Rect(0, 0, 224, 224));

  // a non-square image is cropped, not stretched: bands 0 | 128 | 255 of width
  // 100 -> the output sees the middle band (stretching would give 0 ... 255).
  // Taps near the crop border still read the pixels just outside the crop.
  cv::Mat bands(100, 300, CV_32FC1, cv::Scalar(0));
  bands.colRange(100, 200).setTo(128);
  bands.colRange(200, 300).setTo(255);
  const cv::Mat rb = vlResize(bands, {10, 10});
  CHECK_MSG(std::fabs(rb.at<float>(5, 0) - 112.f) < 1e-3, "%f", rb.at<float>(5, 0));  // 12.5% of taps in band 0
  CHECK(std::fabs(rb.at<float>(5, 5) - 128.f) < 1e-3);
  CHECK_MSG(std::fabs(rb.at<float>(5, 9) - 143.875f) < 1e-3, "%f", rb.at<float>(5, 9));

  // 2x downsampling of a horizontal ramp: centre of output pixel u is 2u + 0.5
  cv::Mat ramp(8, 8, CV_32FC1);
  for (int i = 0; i < 8; ++i) ramp.col(i).setTo(i);
  const cv::Mat rr = vlResize(ramp, {4, 4});
  for (int u = 1; u < 3; ++u) CHECK_MSG(std::fabs(rr.at<float>(0, u) - (2 * u + 0.5f)) < 1e-5, "u=%d %f", u, rr.at<float>(0, u));

  // packChw
  cv::Mat p(2, 2, CV_32FC3);
  for (int i = 0; i < 4; ++i) p.at<cv::Vec3f>(i / 2, i % 2) = cv::Vec3f(float(i), 10.f + i, 20.f + i);
  float chw[12];
  packChw(p, chw);
  CHECK(chw[0] == 0 && chw[3] == 3 && chw[4] == 10 && chw[11] == 23);

  // toRgbFloat: BGR -> RGB and mirror
  cv::Mat bgr(1, 2, CV_8UC3);
  bgr.at<cv::Vec3b>(0, 0) = {1, 2, 3};
  bgr.at<cv::Vec3b>(0, 1) = {4, 5, 6};
  const cv::Mat f = toRgbFloat(bgr, true);
  CHECK(f.at<cv::Vec3f>(0, 0) == cv::Vec3f(6, 5, 4) && f.at<cv::Vec3f>(0, 1) == cv::Vec3f(3, 2, 1));

  return finish("test_preprocess");
}
