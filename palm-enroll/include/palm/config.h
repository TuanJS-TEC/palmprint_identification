#pragma once
#include <string>

#include "palm/capture_gate.h"
#include "palm/palm_model.h"
#include "palm/template.h"

namespace palm {

struct Config {
  ModelOptions model;
  std::string modelVersion = "ee_prnet_e60";
  struct {
    std::string palmOnnx = "palm_detection_mediapipe_2023feb.onnx";
    std::string landmarkOnnx = "handpose_estimation_mediapipe_2023feb.onnx";
  } hand;
  TemplateParams tmpl;
  float dupCosMax = 2.f;     // refuse enrollment if another id matches above; > 1 disables
  float acceptCosMin = 2.f;  // identification accept threshold; > 1 never accepts
  std::string galleryPath = "gallery.db";
  GateLimits gate;           // capture gate (capture_gate.h), config section "capture"
};

// Reads an OpenCV FileStorage YAML file; relative paths are resolved against
// the directory of the config file.
Config loadConfig(const std::string& path);

}  // namespace palm
