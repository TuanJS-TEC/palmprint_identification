#pragma once
// EE-PRnet (models/EE_PRnet-epoch-60.onnx, from convert_mat_to_onnx.py).
// One ONNX graph holds the whole pipeline: ROI-LAnet regresses 9 TPS control
// points from a 56x56 copy of the hand image, the hand image (224x224) is
// warped to a 112x112 palm ROI inside the graph, and the recognition branch
// returns the 512-d fc1R feature. Mean subtraction is also inside the graph:
// inputs are RGB float 0..255.
#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace palm {

struct ModelOptions {
  std::string onnxPath;
  int threads = 4;  // intra-op threads; physical cores
};

struct Embedding {
  std::vector<float> feature;   // raw fc1R output (not normalised)
  std::array<float, 18> theta;  // TPS control points [x1..x9, y1..y9] in [-1, 1]
  cv::Mat roi;                  // CV_32FC3 RGB 112x112, 0..255: what the recogniser saw
};

class PalmModel {
 public:
  explicit PalmModel(const ModelOptions& opt);
  ~PalmModel();
  PalmModel(const PalmModel&) = delete;
  PalmModel& operator=(const PalmModel&) = delete;

  // bgr: hand crop as returned by cv::imread. flip: mirror left-right first
  // (the network was trained on left palms mirrored to look like right palms).
  // The network gets the crop resized the vl_imreadjpeg way to 56x56 and to
  // 224x224 (as in training).
  Embedding run(const cv::Mat& bgr, bool flip);

  int featureDim() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace palm
