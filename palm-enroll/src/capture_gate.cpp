#include "capture_gate.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace palm {
namespace {

// wrist, thumb base, index / middle / ring / pinky knuckles
const int kPalmPts[] = {0, 1, 5, 9, 13, 17};
const int kSample = 64;  // fixed sample grid for the light check: O(1) in the camera resolution

float dist(cv::Point2f a, cv::Point2f b) { return std::hypot(a.x - b.x, a.y - b.y); }

// Every landmark and the palm box keep `margin` (fraction of the frame) from the edge.
bool insideFrame(const HandLandmarks& h, cv::Size size, float margin) {
  const float mx = margin * size.width, my = margin * size.height;
  auto in = [&](float x, float y) { return x >= mx && x <= size.width - mx && y >= my && y <= size.height - my; };
  for (const auto& p : h.pts)
    if (!in(p.x, p.y)) return false;
  return in(h.palmBox.x, h.palmBox.y) && in(h.palmBox.x + h.palmBox.width, h.palmBox.y + h.palmBox.height);
}

// Angle of wrist -> middle knuckle from image up, degrees, 0..180.
float tiltDeg(const HandLandmarks& h) {
  const cv::Point2f v = h.pts[9] - h.pts[0];
  return static_cast<float>(std::abs(std::atan2(v.x, -v.y)) * 180.0 / CV_PI);
}

// Mean length of the four fingers (knuckle -> tip) over the palm length. Curled
// fingers fold towards the palm and get short. Tip-to-tip spread does not work:
// curled fingers still span about one knuckle width.
float fingerOpen(const HandLandmarks& h, float palmLen) {
  float sum = 0.f;
  for (int f = 0; f < 4; ++f) sum += dist(h.pts[5 + 4 * f], h.pts[8 + 4 * f]);  // index, middle, ring, pinky
  return sum / 4 / std::max(1e-5f, palmLen);
}

struct PalmLight {
  float mean = 0.f;     // gray level 0..255
  float satFrac = 0.f;  // share of pixels >= 250
};

// Light measured only inside the palm polygon, so a white wall or a shadow behind
// the hand does not count. The box is resampled to a fixed grid first.
PalmLight palmLight(const cv::Mat& frame, const HandLandmarks& h) {
  std::vector<cv::Point2f> poly;
  for (int k : kPalmPts) poly.push_back(h.pts[k]);
  const cv::Rect box = cv::boundingRect(poly) & cv::Rect(0, 0, frame.cols, frame.rows);
  if (box.empty()) return {};

  cv::Mat small, gray;
  cv::resize(frame(box), small, {kSample, kSample}, 0, 0, cv::INTER_NEAREST);
  cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);

  // fillPoly, not fillConvexPoly: the landmark polygon can be concave with a few px of error
  const float sx = static_cast<float>(kSample) / box.width, sy = static_cast<float>(kSample) / box.height;
  std::vector<cv::Point> pp;
  for (const auto& p : poly) pp.emplace_back(std::lround((p.x - box.x) * sx), std::lround((p.y - box.y) * sy));
  cv::Mat mask = cv::Mat::zeros(kSample, kSample, CV_8U);
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{pp}, cv::Scalar(255));
  const int area = cv::countNonZero(mask);
  if (area == 0) return {};

  PalmLight out;
  out.mean = static_cast<float>(cv::mean(gray, mask)[0]);
  cv::Mat sat;
  cv::compare(gray, 250, sat, cv::CMP_GE);
  cv::bitwise_and(sat, mask, sat);
  out.satFrac = static_cast<float>(cv::countNonZero(sat)) / area;
  return out;
}

}  // namespace

const char* gateMessage(GateStatus s) {
  switch (s) {
    case GateStatus::NoHand: return "Đưa tay phải vào khung hình";
    case GateStatus::LeftHand: return "Đây là tay TRÁI - hãy dùng tay PHẢI";
    case GateStatus::BackOfHand: return "Quay LÒNG bàn tay phải về phía camera";
    case GateStatus::CutOff: return "Đưa cả bàn tay vào giữa khung, không để cắt mép";
    case GateStatus::TooSmall: return "Đưa tay gần hơn";
    case GateStatus::TooLarge: return "Lùi tay ra";
    case GateStatus::BadLight: return "Đổi vị trí hoặc tránh vùng ngược sáng";
    case GateStatus::Rotated: return "Xoay lòng bàn tay thẳng hơn";
    case GateStatus::FingersClosed: return "Mở ngón tay tự nhiên";
    case GateStatus::Blurry: return "Ảnh bị mờ - giữ yên tay";
    case GateStatus::Moving: return "Giữ yên tay";
    case GateStatus::Stabilizing: return "Giữ yên tay...";
    case GateStatus::Ready: return "Đủ điều kiện - đang chụp";
  }
  return "";
}

GateReport CaptureGate::update(const cv::Mat& frame, const HandCheck& chk, const HandLandmarks& h, double nowMs) {
  GateReport r;
  // any failed frame starts the stability window over
  auto fail = [&](GateStatus s) {
    restart();
    r.status = s;
    return r;
  };

  // 1. the hand itself
  if (chk.state == HandState::NoHand) return fail(GateStatus::NoHand);
  if (chk.state == HandState::LeftHand) return fail(GateStatus::LeftHand);
  if (chk.state == HandState::BackOfHand) return fail(GateStatus::BackOfHand);

  // measurements for the log, also on failed frames
  const float palmLen = dist(h.pts[9], h.pts[0]);
  r.palmFrac = palmLen / frame.rows;
  r.angleDeg = tiltDeg(h);
  r.fingerOpen = fingerOpen(h, palmLen);
  const PalmLight light = palmLight(frame, h);
  r.mean = light.mean;
  r.satFrac = light.satFrac;

  // 2. whole hand in the frame (checkHand's OutOfFrame is about the crop box)
  if (chk.state == HandState::OutOfFrame || !insideFrame(h, frame.size(), lim_.edgeMargin)) return fail(GateStatus::CutOff);
  // 3. size: checkHand's crop-side limit or the palm length
  if (chk.state == HandState::TooSmall || r.palmFrac < lim_.minPalmFrac) return fail(GateStatus::TooSmall);
  if (r.palmFrac > lim_.maxPalmFrac) return fail(GateStatus::TooLarge);
  // 4. light
  if (r.mean < lim_.minMean || r.mean > lim_.maxMean || r.satFrac > lim_.maxSatFrac) return fail(GateStatus::BadLight);
  // 5. pose
  if (r.angleDeg > lim_.maxAngleDeg) return fail(GateStatus::Rotated);
  if (r.fingerOpen < lim_.minFingerOpen) return fail(GateStatus::FingersClosed);
  if (chk.state == HandState::Blurry) return fail(GateStatus::Blurry);
  // 6. held still: checkHand's per-frame motion, then the window below
  if (chk.state == HandState::Moving) return fail(GateStatus::Moving);

  // chk.state == Ok: all checks pass on this frame
  if (okSince_ < 0) okSince_ = nowMs;
  ++okFrames_;
  const double elapsed = nowMs - okSince_;
  const float byTime = static_cast<float>(std::min(1.0, elapsed / std::max(1.0, lim_.stableMs)));
  const float byFrames = std::min(1.f, static_cast<float>(okFrames_) / std::max(1, lim_.stableMinFrames));
  r.progress = std::min(byTime, byFrames);
  r.status = r.progress >= 1.f ? GateStatus::Ready : GateStatus::Stabilizing;
  return r;
}

}  // namespace palm
