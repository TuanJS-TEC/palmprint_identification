#pragma once
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

  Embedding run(const cv::Mat& bgr);

  int featureDim() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace palm
