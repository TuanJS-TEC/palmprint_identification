#include "hand_crop.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace palm {

cv::Rect handCropBox(const HandLandmarks& h, float scale) {
  float x0 = h.pts[0].x, y0 = h.pts[0].y, x1 = x0, y1 = y0;
  for (const auto& p : h.pts) {
    x0 = std::min(x0, p.x), y0 = std::min(y0, p.y);
    x1 = std::max(x1, p.x), y1 = std::max(y1, p.y);
  }
  const float side = scale * std::max(x1 - x0, y1 - y0);
  const float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return {static_cast<int>(std::lround(cx - side / 2)), static_cast<int>(std::lround(cy - side / 2)),
          static_cast<int>(std::lround(side)), static_cast<int>(std::lround(side))};
}

float sharpness(const cv::Mat& bgrCrop) {
  cv::Mat g, lap;
  cv::cvtColor(bgrCrop, g, cv::COLOR_BGR2GRAY);
  cv::resize(g, g, {224, 224}, 0, 0, cv::INTER_AREA);
  cv::Laplacian(g, lap, CV_32F);
  cv::Scalar mean, sd;
  cv::meanStdDev(lap, mean, sd);
  return static_cast<float>(sd[0] * sd[0]);
}

float landmarkMotion(const HandLandmarks& a, const HandLandmarks& b, float side) {
  if (!a.found || !b.found || side <= 0) return 1e9f;
  float s = 0;
  for (int k = 0; k < 21; ++k) s += static_cast<float>(cv::norm(a.pts[k] - b.pts[k]));
  return s / 21 / side;
}

const char* stateMessage(HandState s) {
  switch (s) {
    case HandState::NoHand: return "Đưa tay phải vào khung hình";
    case HandState::LeftHand: return "Đây là tay TRÁI - hãy dùng tay PHẢI";
    case HandState::BackOfHand: return "Quay LÒNG bàn tay phải về phía camera";
    case HandState::TooSmall: return "Đưa tay lại gần camera hơn";
    case HandState::OutOfFrame: return "Đưa cả bàn tay vào giữa khung hình";
    case HandState::Moving: return "Giữ yên tay";
    case HandState::Blurry: return "Ảnh bị mờ - giữ yên tay";
    case HandState::Ok: return "Tốt - giữ nguyên";
  }
  return "";
}

HandCheck checkHand(const cv::Mat& frame, const HandLandmarks& h, const HandLandmarks& prev, const CaptureLimits& lim) {
  HandCheck c;
  if (!h.found) return c;
  c.box = handCropBox(h, lim.cropScale);
  // On a non-mirrored frame the detector's label is the physical hand and a
  // palm facing the camera has palmSide() < 0 for the right hand (checked on
  // the capture log with palm_handcheck: 97.6% agreement).
  const bool right = std::string(h.label()) == "Right", palmFront = h.palmSide() < 0;
  if (!right && !palmFront) {
    c.state = HandState::LeftHand;
    return c;
  }
  if (!right || !palmFront) {
    c.state = HandState::BackOfHand;
    return c;
  }
  if (c.box.width < lim.minCropFrac * frame.rows) {
    c.state = HandState::TooSmall;
    return c;
  }
  if ((c.box & cv::Rect(0, 0, frame.cols, frame.rows)) != c.box) {
    c.state = HandState::OutOfFrame;
    return c;
  }
  c.motion = landmarkMotion(h, prev, static_cast<float>(c.box.width));
  if (c.motion > lim.maxMotion) {
    c.state = HandState::Moving;
    return c;
  }
  c.sharp = sharpness(frame(c.box));
  c.state = c.sharp < lim.blurMin ? HandState::Blurry : HandState::Ok;
  return c;
}

}  // namespace palm
