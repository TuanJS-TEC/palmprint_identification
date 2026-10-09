#include "config.h"

#include <filesystem>
#include <stdexcept>

#include <opencv2/core/persistence.hpp>

namespace palm {
namespace {

template <typename T>
void read(const cv::FileNode& n, const char* key, T& v) {
  if (!n[key].empty()) n[key] >> v;
}

std::string resolve(const std::filesystem::path& base, const std::string& p) {
  const std::filesystem::path q(p);
  return (q.is_absolute() ? q : base / q).lexically_normal().string();
}

}  // namespace

Config loadConfig(const std::string& path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) throw std::runtime_error("cannot open config: " + path);
  const auto base = std::filesystem::absolute(path).parent_path();
  Config c;

  const cv::FileNode m = fs["model"];
  read(m, "onnx", c.model.onnxPath);
  read(m, "threads", c.model.threads);
  read(m, "version", c.modelVersion);
  std::string mode = "resize224";
  read(m, "input224_mode", mode);
  if (mode != "resize224" && mode != "original") throw std::runtime_error("model.input224_mode: resize224 | original");
  c.model.input224Original = mode == "original";
  c.model.onnxPath = resolve(base, c.model.onnxPath);

  read(fs["hand"], "right_label", c.rightLabel);
  read(fs["hand"], "palm_onnx", c.hand.palmOnnx);
  read(fs["hand"], "landmark_onnx", c.hand.landmarkOnnx);
  c.hand.palmOnnx = resolve(base, c.hand.palmOnnx);
  c.hand.landmarkOnnx = resolve(base, c.hand.landmarkOnnx);
  if (c.rightLabel != "Left" && c.rightLabel != "Right") throw std::runtime_error("hand.right_label: Left | Right");
  int mirror = 0;
  read(fs["preprocess"], "mirror_input", mirror);
  c.mirrorInput = mirror != 0;

  const cv::FileNode t = fs["template"];
  read(t, "min_valid", c.tmpl.minValid);
  read(t, "outlier_cos_min", c.tmpl.outlierCosMin);
  read(t, "dup_cos_max", c.dupCosMax);

  read(fs["match"], "accept_cos_min", c.acceptCosMin);

  const cv::FileNode cam = fs["camera"];
  read(cam, "device", c.camera.device);
  read(cam, "width", c.camera.width);
  read(cam, "height", c.camera.height);
  int displayMirror = c.camera.displayMirror;
  read(cam, "display_mirror", displayMirror);
  c.camera.displayMirror = displayMirror != 0;
  const cv::FileNode cap = fs["capture"];
  read(cap, "crop_scale", c.capture.cropScale);
  read(cap, "min_crop_frac", c.capture.minCropFrac);
  read(cap, "max_motion", c.capture.maxMotion);
  read(cap, "blur_min", c.capture.blurMin);
  read(cap, "edge_margin", c.gate.edgeMargin);
  read(cap, "min_palm_frac", c.gate.minPalmFrac);
  read(cap, "max_palm_frac", c.gate.maxPalmFrac);
  read(cap, "max_angle_deg", c.gate.maxAngleDeg);
  read(cap, "max_tilt_deg", c.gate.maxTiltDeg);
  read(cap, "frontal_max_deg", c.gate.frontalMaxDeg);
  read(cap, "min_pose_sep_deg", c.gate.minPoseSepDeg);
  read(cap, "min_finger_open", c.gate.minFingerOpen);
  read(cap, "min_mean", c.gate.minMean);
  read(cap, "max_mean", c.gate.maxMean);
  read(cap, "max_sat_frac", c.gate.maxSatFrac);
  read(cap, "stable_ms", c.gate.stableMs);
  read(cap, "stable_min_frames", c.gate.stableMinFrames);
  const cv::FileNode ses = fs["session"];
  read(ses, "enroll_samples", c.session.samples);
  read(ses, "enroll_gap_ms", c.session.gapMs);
  read(ses, "enroll_timeout_s", c.session.timeoutS);
  read(ses, "identify_confirm_frames", c.session.identifyConfirm);
  read(fs.root(), "gallery", c.galleryPath);
  c.galleryPath = resolve(base, c.galleryPath);
  return c;
}

}  // namespace palm
