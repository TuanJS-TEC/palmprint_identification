#pragma once
#include <QColor>
#include <QString>
#include <QWidget>

#include "camera_worker.h"

// Camera preview with the hand overlay (crop box, landmarks, check message).
class VideoView : public QWidget {
  Q_OBJECT
 public:
  // lim: scale of the enrollment tilt map
  VideoView(bool mirror, const palm::GateLimits& lim, QWidget* parent = nullptr);
  void setFrame(const FrameResult& r);
  void setMessage(const QString& text) { message_ = text; update(); }  // shown when no frame
  void setBanner(const QString& text, const QColor& color);            // session status, bottom

 protected:
  void paintEvent(QPaintEvent*) override;

 private:
  FrameResult r_;
  bool mirror_;
  palm::GateLimits lim_;
  QString message_, banner_;
  QColor bannerColor_;
};
