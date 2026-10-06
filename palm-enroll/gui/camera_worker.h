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

#include "config.h"
#include "hand_crop.h"
#include "hand_detector.h"

struct FrameResult {
  QImage view;               // RGB888 camera frame (not mirrored), scaled for display
  double viewScale = 1.0;    // view pixels per camera pixel
  palm::HandLandmarks hand;  // camera pixels
  palm::HandCheck check;
  bool embedded = false;
  std::vector<float> feat;   // L2-normalised fc1R feature
  QImage roi;                // 112x112 aligned palm ROI the recogniser saw
  QImage crop;               // hand crop given to EE-PRnet (thumbnail)
  double fps = 0, msDetect = 0, msEmbed = 0;
};
Q_DECLARE_METATYPE(FrameResult)

class CameraWorker : public QObject {
  Q_OBJECT
 public:
  // source: empty = camera cfg.camera.device; else a video file or an image
  // (an image is replayed as a still camera; mirrorSource un-mirrors it).
  CameraWorker(const palm::Config& cfg, std::string source, bool mirrorSource);
  ~CameraWorker() override;

  void stop() { stop_ = true; }
  // -1: no embeddings, 0: every frame that passes the checks, > 0: at most one per interval
  void setEmbedInterval(int ms) { embedIntervalMs_ = ms; }
  void setVerbose(bool v) { verbose_ = v; }  // timing summary on stderr every second

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
  std::atomic<int> embedIntervalMs_{-1};
  std::atomic<bool> verbose_{false};
};
