#pragma once
// MediaPipe hand pipeline (OpenCV Zoo ONNX conversions, 2023feb):
//   palm_detection_mediapipe_2023feb.onnx      -> palm box + 7 palm keypoints
//   handpose_estimation_mediapipe_2023feb.onnx -> 21 landmarks + handedness
// Port of opencv_zoo models/{palm_detection,handpose_estimation}_mediapipe/
// mp_palmdet.py and mp_handpose.py, run with ONNX Runtime.
#include <array>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace palm {

struct HandLandmarks {
  bool found = false;
  float palmScore = 0.f;   // palm detector score
  float presence = 0.f;    // landmark model hand presence
  float handedness = 0.f;  // model output, > 0.5 means MediaPipe label "Right"
  cv::Rect2f palmBox;      // palm detector box, pixels
  std::array<cv::Point2f, 21> pts{};  // landmarks, pixels of the input image

  // How the hand looks in the given image: on a non-mirrored camera frame this
  // is the physical hand (palm_handcheck on the capture log). The capture
  // script mirrored its frames first, so log_label.csv has the opposite label.
  const char* label() const { return handedness > 0.5f ? "Right" : "Left"; }
  // z of cross(index_mcp - wrist, pinky_mcp - wrist) in image coordinates:
  // < 0 for a right palm or a left back of hand facing a non-mirrored camera.
  float palmSide() const;
};

class HandDetector {
 public:
  HandDetector(const std::string& palmOnnx, const std::string& landmarkOnnx, int threads = 2);
  ~HandDetector();
  HandDetector(const HandDetector&) = delete;
  HandDetector& operator=(const HandDetector&) = delete;

  // bgr: full camera frame. Returns the hand with the largest palm box.
  HandLandmarks detect(const cv::Mat& bgr);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace palm
