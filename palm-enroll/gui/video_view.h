#pragma once
#include <QColor>
#include <QString>
#include <QWidget>

#include "camera_worker.h"

// Camera preview with the hand overlay (crop box, landmarks, check message).
class VideoView : public QWidget {
  Q_OBJECT
 public:
  explicit VideoView(bool mirror, QWidget* parent = nullptr);
  void setFrame(const FrameResult& r);
  void setMessage(const QString& text) { message_ = text; update(); }  // shown when no frame
  void setBanner(const QString& text, const QColor& color);            // session status, bottom

 protected:
  void paintEvent(QPaintEvent*) override;

 private:
  FrameResult r_;
  bool mirror_;
  QString message_, banner_;
  QColor bannerColor_;
};
