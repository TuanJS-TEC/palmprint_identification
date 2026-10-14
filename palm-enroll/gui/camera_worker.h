#pragma once
// Capture + processing thread of palm_gui: grabs camera frames, runs the hand
// detector and the live checks on every frame and, when asked, the EE-PRnet
// embedding on frames that pass the checks.
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <QImage>
#include <QMetaType>
#include <QObject>

#include "capture_gate.h"
#include "config.h"
#include "hand_detector.h"

struct FrameResult {
  QImage view;               // RGB888 camera frame (not mirrored), scaled for display
  double viewScale = 1.0;    // view pixels per camera pixel
  palm::HandLandmarks hand;  // camera pixels
  bool gated = false;        // enrollment: Ready of the capture gate decides; else every frame it passes
  palm::GateReport gate;     // camera pixels
  std::vector<cv::Vec3f> taken;  // enrollment: palm normals of the samples taken so far
  std::vector<float> feat;   // L2-normalised fc1R feature; empty if no embedding on this frame
  QImage roi;                // 112x112 aligned palm ROI the recogniser saw
  QImage crop;               // hand crop given to EE-PRnet (thumbnail)
  double fps = 0, msDetect = 0, msEmbed = 0;

  bool embedded() const { return !feat.empty(); }
};
Q_DECLARE_METATYPE(FrameResult)

class CameraWorker : public QObject {
  Q_OBJECT
 public:
  // Idle: no embeddings. Identify: an embedding on every frame the capture
  // gate passes. Enroll: only on frames it reports Ready, each at a palm tilt
  // not taken yet, at most one per session.gapMs.
  enum class Mode { Idle, Identify, Enroll };

  // source: empty = camera cfg.camera.device; else a video file or an image
  // (an image is replayed as a still camera; mirrorSource un-mirrors it).
  CameraWorker(const palm::Config& cfg, std::string source, bool mirrorSource);
  ~CameraWorker() override;

  void stop() { stop_ = true; }
  void setMode(Mode m) { mode_ = m; }

 public slots:
  void run();  // blocking loop until stop()

 signals:
  void status(const QString& text);
  void failed(const QString& text);
  void frameReady(const FrameResult& r);

 private:
  palm::Config cfg_;
  std::string source_;
  bool mirrorSource_;
  std::atomic<bool> stop_{false};
  std::atomic<Mode> mode_{Mode::Idle};
};
