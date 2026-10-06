#pragma once
#include <QElapsedTimer>
#include <QMainWindow>
#include <QThread>

#include "camera_worker.h"
#include "config.h"
#include "gallery.h"

class QLabel;
class QLineEdit;
class QListWidget;
class QProgressBar;
class QPushButton;
class VideoView;

// Right-hand-only enrollment and 1:N identification on a live camera.
class MainWindow : public QMainWindow {
  Q_OBJECT
 public:
  MainWindow(const palm::Config& cfg, const std::string& source, bool mirrorSource);
  ~MainWindow() override;

  void beginEnroll(const QString& id);  // as if typed and clicked
  void beginIdentify();
  void setVerbose(bool v) { worker_->setVerbose(v); }

 protected:
  void keyPressEvent(QKeyEvent* e) override;

 private slots:
  void onFrame(const FrameResult& r);
  void onWorkerStatus(const QString& text);
  void onWorkerFailed(const QString& text);
  void startEnroll();
  void cancelEnroll();
  void toggleIdentify(bool on);
  void deleteSelected();

 private:
  enum class Mode { Idle, Enroll, Identify };
  void setMode(Mode m);
  void addEnrollSample(const FrameResult& r);
  void finishEnroll();
  void identify(const FrameResult& r);
  void refreshList();
  QString nextFreeId() const;

  palm::Config cfg_;
  palm::Gallery gallery_;
  Mode mode_ = Mode::Idle;

  QThread thread_;
  CameraWorker* worker_ = nullptr;

  // enrollment session
  QString enrollId_;
  std::vector<palm::Vec> enrollFeats_;
  QElapsedTimer enrollClock_;
  // identification
  QString lastId_;
  int streak_ = 0;
  QElapsedTimer lastEmbedClock_;

  VideoView* video_;
  QLabel* status_;
  QLineEdit* idEdit_;
  QPushButton *enrollBtn_, *cancelBtn_, *identifyBtn_, *deleteBtn_;
  QProgressBar* progress_;
  std::vector<QLabel*> thumbs_;
  QLabel* enrollMsg_;
  QLabel* result_;
  QLabel *roiView_, *cropView_;
  QListWidget* list_;
};
