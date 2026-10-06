// Capture gate on synthetic hands: no ONNX models needed. The frames are flat
// gray, so checkHand's blur check passes (blurMin 0) and the light check sees
// exactly the gray level.
#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include "capture_gate.h"
#include "check.h"

using namespace palm;

namespace {

const int kW = 1920, kH = 1080;

// Right palm facing the camera, wrist at (wristX, wristY), palm length L (wrist ->
// middle knuckle). Palm up = -y. openScale 1: fingers straight, 0.3: curled.
// tiltDeg rotates the whole hand around the wrist.
HandLandmarks makeHand(float wristX, float wristY, float L, float openScale = 1.f, float tiltDeg = 0.f) {
  const float W = 0.5f * L;
  std::array<cv::Point2f, 21> p{};
  p[0] = {0, 0};
  p[1] = {0.8f * W, -0.3f * L};
  p[2] = {1.0f * W, -0.5f * L};
  p[3] = {1.1f * W, -0.65f * L};
  p[4] = {1.2f * W, -0.8f * L};
  const cv::Point2f mcp[4] = {{0.5f * W, -0.9f * L}, {0, -L}, {-W / 3, -0.95f * L}, {-0.5f * W, -0.85f * L}};
  const float ox[4] = {1.f, 0.3f, -0.3f, -1.f};  // index .. pinky, outward lean
  for (int f = 0; f < 4; ++f) {
    const float tx = ox[f] * 0.1f * L, ty = -0.8f * L * openScale;  // knuckle -> tip
    const int b = 5 + 4 * f;
    p[b] = mcp[f];
    p[b + 1] = {mcp[f].x + tx / 3, mcp[f].y + ty / 3};
    p[b + 2] = {mcp[f].x + 2 * tx / 3, mcp[f].y + 2 * ty / 3};
    p[b + 3] = {mcp[f].x + tx, mcp[f].y + ty};
  }
  const float a = tiltDeg * static_cast<float>(CV_PI / 180.0);
  HandLandmarks h;
  h.found = true;
  h.palmScore = 1.f;
  h.presence = 1.f;
  h.handedness = 0.9f;  // "Right" on a non-mirrored frame
  float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
  for (int i = 0; i < 21; ++i) {
    const float x = p[i].x * std::cos(a) - p[i].y * std::sin(a) + wristX;
    const float y = p[i].x * std::sin(a) + p[i].y * std::cos(a) + wristY;
    h.pts[i] = {x, y};
    x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
  }
  h.palmBox = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
  return h;
}

// One frame through checkHand and the gate, as the GUI does it.
GateReport step(CaptureGate& g, const cv::Mat& frame, const HandLandmarks& h, const HandLandmarks& prev, double tMs) {
  return g.update(frame, checkHand(frame, h, prev, CaptureLimits{}), h, tMs);
}

}  // namespace

int main() {
  const cv::Mat gray(kH, kW, CV_8UC3, cv::Scalar::all(120));
  const float L = 0.22f * kH;  // palm length 22% of the frame height
  const HandLandmarks good = makeHand(960, 900, L);

  // every status has a message
  for (int i = 0; i <= static_cast<int>(GateStatus::Ready); ++i)
    CHECK(std::string(gateMessage(static_cast<GateStatus>(i))).size() > 0);

  // measurements on a good hand
  {
    CaptureGate g;
    const GateReport r = step(g, gray, good, good, 0);
    CHECK(r.status == GateStatus::Stabilizing);
    CHECK(std::fabs(r.palmFrac - 0.22f) < 1e-3f);
    CHECK(std::fabs(r.angleDeg) < 0.5f);
    CHECK(r.fingerOpen > 0.6f);
    CHECK(std::fabs(r.mean - 120.f) < 1.f);
    CHECK(r.satFrac == 0.f);
  }

  // stability: Ready only once 400 ms AND 10 frames have passed, both at 10 ms steps
  {
    CaptureGate g;
    for (int t = 0; t <= 390; t += 10) {
      const GateReport r = step(g, gray, good, good, t);
      CHECK_MSG(r.status == GateStatus::Stabilizing, "t=%d", t);
    }
    CHECK(step(g, gray, good, good, 400).status == GateStatus::Ready);
  }
  // slow machine: 100 ms per frame, the frame count is the limit
  {
    CaptureGate g;
    for (int t = 0; t <= 800; t += 100) CHECK(step(g, gray, good, good, t).status == GateStatus::Stabilizing);
    CHECK(step(g, gray, good, good, 900).status == GateStatus::Ready);
  }
  // progress grows with the window (min of time and frame count)
  {
    CaptureGate g;
    float p200 = 0;
    for (int t = 0; t <= 200; t += 10) p200 = step(g, gray, good, good, t).progress;
    CHECK(std::fabs(p200 - 0.5f) < 0.01f);
  }
  // restart after a capture: the next window starts from zero
  {
    CaptureGate g;
    for (int t = 0; t <= 400; t += 10) step(g, gray, good, good, t);
    g.restart();
    const GateReport r = step(g, gray, good, good, 500);
    CHECK(r.status == GateStatus::Stabilizing && r.progress == 0.f);
  }
  // a failed frame in the middle resets the window
  {
    CaptureGate g;
    for (int t = 0; t <= 300; t += 10) step(g, gray, good, good, t);
    const cv::Mat dark(kH, kW, CV_8UC3, cv::Scalar::all(20));
    CHECK(step(g, dark, good, good, 310).status == GateStatus::BadLight);
    const GateReport r = step(g, gray, good, good, 320);
    CHECK(r.status == GateStatus::Stabilizing && r.progress == 0.f);
  }

  // failures, in the order of the gate; each on a fresh gate
  auto once = [](const cv::Mat& frame, const HandLandmarks& h, const HandLandmarks& prev) {
    CaptureGate g;
    return step(g, frame, h, prev, 0).status;
  };
  CHECK(once(gray, HandLandmarks{}, good) == GateStatus::NoHand);
  {
    HandLandmarks back = good;
    back.handedness = 0.1f;  // checkHand: label Left with a palm facing: back of the hand
    CHECK(once(gray, back, back) == GateStatus::BackOfHand);
  }
  CHECK(once(gray, makeHand(960, 1060, L), makeHand(960, 1060, L)) == GateStatus::CutOff);
  CHECK(once(gray, makeHand(960, 900, 0.10f * kH), makeHand(960, 900, 0.10f * kH)) == GateStatus::TooSmall);
  CHECK(once(gray, makeHand(960, 900, 0.36f * kH), makeHand(960, 900, 0.36f * kH)) == GateStatus::TooLarge);
  {
    const cv::Mat dark(kH, kW, CV_8UC3, cv::Scalar::all(20));
    CHECK(once(dark, good, good) == GateStatus::BadLight);
    const cv::Mat bright(kH, kW, CV_8UC3, cv::Scalar::all(255));
    CHECK(once(bright, good, good) == GateStatus::BadLight);
  }
  {
    const HandLandmarks tilted = makeHand(960, 900, L, 1.f, 50.f);
    CaptureGate g;
    const GateReport r = step(g, gray, tilted, tilted, 0);
    CHECK(r.status == GateStatus::Rotated);
    CHECK(std::fabs(r.angleDeg - 50.f) < 1.f);
  }
  // curled fingers make the bounding box smaller, so the palm is a bit larger here
  const HandLandmarks curled = makeHand(960, 900, 0.26f * kH, 0.3f);
  CHECK(once(gray, curled, curled) == GateStatus::FingersClosed);
  CHECK(once(gray, good, makeHand(990, 900, L)) == GateStatus::Moving);

  return finish("capture_gate");
}
