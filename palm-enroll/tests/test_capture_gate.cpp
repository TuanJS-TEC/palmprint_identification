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
// tiltDeg rotates the whole hand around the wrist (in the image plane). The
// world landmarks are the same layout in metres, tilted out of the image plane
// by outDeg around an image-plane axis at outAxisDeg (0: the x axis).
HandLandmarks makeHand(float wristX, float wristY, float L, float openScale = 1.f, float tiltDeg = 0.f,
                       float outDeg = 0.f, float outAxisDeg = 0.f) {
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
  const float phi = outAxisDeg * static_cast<float>(CV_PI / 180.0);
  const cv::Vec3f k(std::cos(phi), std::sin(phi), 0.f);  // unit rotation axis
  const float th = outDeg * static_cast<float>(CV_PI / 180.0);
  const float m = 0.09f / L;  // a palm is ~9 cm long
  for (int i = 0; i < 21; ++i) {
    const cv::Vec3f q((h.pts[i].x - wristX) * m, (h.pts[i].y - wristY) * m, 0.f);
    // Rodrigues' rotation formula
    const cv::Vec3f w = q * std::cos(th) + k.cross(q) * std::sin(th) + k * (k.dot(q) * (1 - std::cos(th)));
    h.world[i] = {w[0], w[1], w[2]};
  }
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
    CHECK(r.tiltDeg < 0.5f && r.normal[2] < -0.99f && r.poseSepDeg < 0.f);
    CHECK(r.framePasses());
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

  // out-of-plane tilt: measured in any direction, refused above maxTiltDeg (25)
  for (float axis : {0.f, 90.f, 45.f, 200.f}) {
    const HandLandmarks t20 = makeHand(960, 900, L, 1.f, 0.f, 20.f, axis);
    CaptureGate g;
    const GateReport r = step(g, gray, t20, t20, 0);
    CHECK_MSG(r.status == GateStatus::Stabilizing && std::fabs(r.tiltDeg - 20.f) < 0.5f, "axis %.0f", axis);
    CHECK(r.normal[2] < 0.f && std::fabs(cv::norm(r.normal) - 1.0) < 1e-4);
    const HandLandmarks t30 = makeHand(960, 900, L, 1.f, 0.f, 30.f, axis);
    CHECK_MSG(once(gray, t30, t30) == GateStatus::Tilted, "axis %.0f", axis);
  }
  // the in-plane rotation does not count as tilt
  {
    const HandLandmarks rot = makeHand(960, 900, L, 1.f, 30.f);
    CaptureGate g;
    const GateReport r = step(g, gray, rot, rot, 0);
    CHECK(r.status == GateStatus::Stabilizing && r.tiltDeg < 0.5f && std::fabs(r.angleDeg - 30.f) < 1.f);
  }
  // no world landmarks (degenerate): no plane, refused as tilted
  {
    HandLandmarks flat = good;
    flat.world = {};
    CHECK(cv::norm(palmNormal(flat)) == 0.0);
    CHECK(once(gray, flat, flat) == GateStatus::Tilted);
  }
  CHECK(std::fabs(normalAngleDeg({0, 0, -1}, {0, 0, -1})) < 1e-3f);
  CHECK(std::fabs(normalAngleDeg({0, 0, -1}, {1, 0, 0}) - 90.f) < 1e-3f);
  CHECK(normalAngleDeg({0, 0, 0}, {0, 0, -1}) == 90.f);

  // enrollment pose coverage
  {
    CaptureGate g;
    g.setCollecting(true);
    const HandLandmarks t20 = makeHand(960, 900, L, 1.f, 0.f, 20.f, 0.f);
    CHECK(step(g, gray, t20, t20, 0).status == GateStatus::NotFrontal);  // the first sample faces the camera
    const HandLandmarks front = makeHand(960, 900, L, 1.f, 0.f, 5.f, 0.f);
    GateReport r = step(g, gray, front, front, 10);
    CHECK(r.status == GateStatus::Stabilizing && r.poseSepDeg < 0.f);
    g.addTaken(r.normal);
    g.restart();
    // same tilt again, or within minPoseSepDeg (8) of it: refused
    CHECK(step(g, gray, front, front, 20).status == GateStatus::SamePose);
    const HandLandmarks near = makeHand(960, 900, L, 1.f, 0.f, 10.f, 0.f);
    r = step(g, gray, near, near, 30);
    CHECK(r.status == GateStatus::SamePose && std::fabs(r.poseSepDeg - 5.f) < 0.5f);
    // 20 degrees the other way: 25 degrees from the first sample, taken
    const HandLandmarks other = makeHand(960, 900, L, 1.f, 0.f, 20.f, 180.f);
    r = step(g, gray, other, other, 40);
    CHECK(r.status == GateStatus::Stabilizing && std::fabs(r.poseSepDeg - 25.f) < 0.5f);
    g.addTaken(r.normal);
    CHECK(g.taken().size() == 2);
    // the closest taken sample decides: 15 degrees about the y axis is ~16 from the first, 30+ from the second
    const HandLandmarks side = makeHand(960, 900, L, 1.f, 0.f, 15.f, 90.f);
    r = step(g, gray, side, side, 50);
    CHECK(r.status == GateStatus::Stabilizing && r.poseSepDeg > 14.f && r.poseSepDeg < 17.f);
    // still too tilted overall: Tilted comes before the coverage checks
    const HandLandmarks far = makeHand(960, 900, L, 1.f, 0.f, 30.f, 90.f);
    CHECK(step(g, gray, far, far, 60).status == GateStatus::Tilted);
    // switching off (identification) forgets the samples
    g.setCollecting(false);
    CHECK(g.taken().empty() && step(g, gray, front, front, 70).status == GateStatus::Stabilizing);
    CHECK(step(g, gray, t20, t20, 80).status == GateStatus::Stabilizing);  // no frontal rule either
  }

  return finish("capture_gate");
}
