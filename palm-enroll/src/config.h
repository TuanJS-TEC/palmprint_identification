#pragma once
#include <string>

#include "capture_gate.h"
#include "gallery.h"
#include "hand_crop.h"
#include "palm_model.h"

namespace palm {

struct Config {
  ModelOptions model;
  std::string modelVersion = "ee_prnet_e60";
  // Only the physical right hand is enrolled and identified. MediaPipe labels
  // handedness assuming a mirrored (selfie) image, so which label means the
  // physical right hand depends on the capture: "Right" for mirrored frames,
  // "Left" for non-mirrored frames.
  std::string rightLabel = "Left";
  struct {
    std::string palmOnnx = "palm_detection_mediapipe_2023feb.onnx";
    std::string landmarkOnnx = "handpose_estimation_mediapipe_2023feb.onnx";
  } hand;
  // The network expects a right palm seen from the front, not mirrored (thumb
  // on the image's right). Set when the hand crops are mirror images.
  bool mirrorInput = false;
  TemplateParams tmpl;
  float dupCosMax = 2.f;     // refuse enrollment if another id matches above; > 1 disables
  float acceptCosMin = 2.f;  // identification accept threshold; > 1 never accepts
  std::string galleryPath = "gallery.bin";

  // live capture (palm_gui)
  struct {
    int device = 0, width = 1920, height = 1080;
    bool displayMirror = true;  // show the preview like a mirror
  } camera;
  CaptureLimits capture;
  GateLimits gate;            // enrollment capture gate (capture_gate.h)
  struct {
    int samples = 5;            // embeddings per enrollment
    int gapMs = 700;            // between two enrollment captures
    int timeoutS = 60;
    int identifyConfirm = 3;    // consecutive accepted frames to confirm an id
  } session;

  bool isRightHand(const std::string& mediapipeLabel) const { return mediapipeLabel == rightLabel; }
};

// Reads an OpenCV FileStorage YAML file; relative paths are resolved against
// the directory of the config file.
Config loadConfig(const std::string& path);

}  // namespace palm
