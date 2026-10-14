#pragma once
#include <QElapsedTimer>
#include <QMainWindow>
#include <QThread>

#include "camera_worker.h"
#include "config.h"
#include "event_log.h"
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
  // logPath: event log file (empty: log panel only)
  MainWindow(const palm::Config& cfg, const std::string& source, bool mirrorSource, const QString& logPath);
  ~MainWindow() override;

  void beginEnroll(const QString& id);  // as if typed and clicked
  void beginIdentify();

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
  using Mode = CameraWorker::Mode;
  void setMode(Mode m);
  void addEnrollSample(const FrameResult& r);
  void finishEnroll();
  void identify(const FrameResult& r);
  void refreshList();
  void logFrame(const FrameResult& r);  // hand state changes + periodic timing summary

  palm::Config cfg_;
  palm::Gallery gallery_;
  bool galleryOk_ = true;  // false: the gallery file could not be opened and must not be overwritten
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
  // log
  int loggedState_ = -1;  // palm::GateStatus of the last HAND line
  QElapsedTimer handLogClock_, perfClock_;
  int perfFrames_ = 0, perfEmbeds_ = 0;
  double perfDetectMs_ = 0, perfEmbedMs_ = 0;

  VideoView* video_;
  QLabel* status_;
  QLineEdit* idEdit_;
  QPushButton *enrollBtn_, *cancelBtn_, *identifyBtn_, *deleteBtn_, *logBtn_;
  QProgressBar* progress_;
  std::vector<QLabel*> thumbs_;
  QLabel* enrollMsg_;
  QLabel* result_;
  QLabel *roiView_, *cropView_;
  QLabel *debugView_, *debugInfo_;
  QListWidget* list_;
  EventLog* log_;
};
