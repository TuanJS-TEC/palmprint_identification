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
  int threads = 4;               // intra-op threads; physical cores
  bool input224Original = false; // false: Input224 = 224x224 as in training
                                 // true : Input224 = original size (buildFeat 'imgOrg')
};

struct ModelInputs {
  cv::Mat x56;   // CV_32FC3 RGB 56x56
  cv::Mat x224;  // CV_32FC3 RGB 224x224 (or original size)
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
  ModelInputs preprocess(const cv::Mat& bgr, bool flip) const;
  Embedding infer(const ModelInputs& in);
  Embedding run(const cv::Mat& bgr, bool flip) { return infer(preprocess(bgr, flip)); }

  int featureDim() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  bool input224Original_;
};

}  // namespace palm
