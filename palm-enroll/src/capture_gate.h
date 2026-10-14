#pragma once
// Capture gate of a live camera: decides when a hand is good enough to take an
// embedding. Plain C++ and OpenCV only (no GUI, no Qt).
#include <vector>

#include <opencv2/core.hpp>

#include "hand_detector.h"

namespace palm {

// Order of the checks is the order of the user's guidance: the hand must be
// visible and whole, then its size, light, pose, (enrollment) a tilt not
// covered yet, and finally held still.
enum class GateStatus {
  NoHand,
  LeftHand,
  BackOfHand,
  CutOff,       // wrist, palm or finger roots leave the frame
  TooSmall,     // palm too small in the frame: move closer
  TooLarge,     // palm too large in the frame: move back
  BadLight,     // palm too dark, blown out or saturated: change place / avoid backlight
  Rotated,      // wrist -> middle knuckle turned too far from upright (in the image plane)
  Tilted,       // palm plane tilted too far from the image plane (out of plane)
  FingersClosed,
  Blurry,
  NotFrontal,   // enrollment, first sample: the palm must face the camera
  SamePose,     // enrollment: too close to the tilt of a sample already taken
  Moving,       // hand still moving between frames
  Stabilizing,  // good so far: hold still until the window is full
  Ready,        // capture now
};

const char* gateMessage(GateStatus s);  // Vietnamese UI text

struct GateLimits {
  float cropScale = 1.17f;    // hand crop side / landmark box (see handCropBox in capture_gate.cpp)
  float minCropFrac = 0.33f;  // crop side / frame height
  float maxMotion = 0.02f;    // landmark motion per frame, relative to the crop side
  float blurMin = 0.f;        // Laplacian variance of the crop
  float edgeMargin = 0.02f;   // landmarks keep this fraction of the frame size from the edge
  float minPalmFrac = 0.15f;  // wrist -> middle knuckle / frame height (estimate, calibrate with the log)
  float maxPalmFrac = 0.30f;
  float maxAngleDeg = 35.f;   // wrist -> middle knuckle vs image up
  // Out-of-plane tilt of the palm, from the world landmarks (palmNormal). The
  // EE-PRnet embedding of one palm changes with it: same-person cos median 0.98
  // within 5 degrees, 0.75 at 15-20 degrees (output/original, 2026-10-09).
  float maxTiltDeg = 25.f;     // every frame, enrollment and identification
  float frontalMaxDeg = 12.f;  // enrollment: the first sample
  float minPoseSepDeg = 8.f;   // enrollment: each later sample vs every earlier one (noise ~2-5 degrees)
  float minFingerOpen = 0.6f; // mean knuckle -> tip of the four fingers / wrist -> middle knuckle
  float minMean = 50.f;       // palm gray level 0..255
  float maxMean = 200.f;
  float maxSatFrac = 0.05f;   // share of palm pixels >= 250
  double stableMs = 400;      // hand held still for at least this long ...
  int stableMinFrames = 10;   // ... and over at least this many frames
};

// What the gate measured on the last frame; logged with each sample.
struct GateReport {
  GateStatus status = GateStatus::NoHand;
  float progress = 0.f;   // 0..1 of the stability window, for the UI ring
  cv::Rect box;           // hand crop box in the frame (what EE-PRnet gets); empty without a hand
  float motion = 0.f;     // see GateLimits::maxMotion; 1e9 without a hand in the previous frame, 0 if not measured
  float sharp = 0.f;      // see GateLimits::blurMin; 0 if not measured
  float palmFrac = 0.f;   // wrist -> middle knuckle / frame height
  float angleDeg = 0.f;   // in-plane rotation from upright
  float tiltDeg = 0.f;    // out-of-plane tilt of the palm plane
  cv::Vec3f normal;       // palmNormal(); zero if not measured
  float poseSepDeg = -1.f; // enrollment: angle to the closest sample taken, -1 if none
  float fingerOpen = 0.f; // see GateLimits::minFingerOpen
  float mean = 0.f;       // palm gray level
  float satFrac = 0.f;    // saturated palm pixels
  cv::Mat lightDebug;     // BGR 64x64: palm pixels of the light check, saturated ones red; empty if not measured

  // Every check passed on this frame (identification needs no stable window).
  bool framePasses() const { return status == GateStatus::Stabilizing || status == GateStatus::Ready; }
};

class CaptureGate {
 public:
  explicit CaptureGate(const GateLimits& lim = {}) : lim_(lim) {}

  // frame: non-mirrored camera frame (what the detector saw); h: its
  // detection; nowMs: monotonic clock in milliseconds. Call on every frame:
  // the motion check compares with the previous call.
  GateReport update(const cv::Mat& frame, const HandLandmarks& h, double nowMs);

  // Starts a new stability window with the next frame (after each capture).
  void restart() { okSince_ = -1; okFrames_ = 0; }

  // Enrollment pose coverage. While collecting, the first sample must face the
  // camera (frontalMaxDeg) and every later one must differ from all samples
  // taken so far by minPoseSepDeg, so the stored samples cover the tilts a
  // probe can have. Switching it on or off forgets the samples taken.
  void setCollecting(bool on) {
    collecting_ = on;
    taken_.clear();
    restart();
  }
  bool collecting() const { return collecting_; }
  // The palm normal (GateReport::normal) of a sample just taken.
  void addTaken(const cv::Vec3f& normal) { taken_.push_back(normal); }
  const std::vector<cv::Vec3f>& taken() const { return taken_; }

 private:
  GateLimits lim_;
  HandLandmarks prev_;  // detection of the previous frame (motion)
  bool collecting_ = false;
  std::vector<cv::Vec3f> taken_;
  double okSince_ = -1;  // start of the current run of good frames, -1 if none
  int okFrames_ = 0;
};

}  // namespace palm
