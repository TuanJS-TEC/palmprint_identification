#include "main_window.h"

#include <algorithm>
#include <filesystem>

#include <QApplication>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>

#include "video_view.h"

namespace {

const QColor kGreen(39, 174, 96), kRed(192, 57, 43), kAmber(211, 140, 20), kBlue(41, 128, 185);

QLabel* imageBox(int side) {
  auto* l = new QLabel;
  l->setFixedSize(side, side);
  l->setAlignment(Qt::AlignCenter);
  l->setStyleSheet("background:#222; border:1px solid #555; color:#888;");
  return l;
}

QPixmap pix(const QImage& img, int side) {
  return QPixmap::fromImage(img).scaled(side, side, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

QString resultHtml(const QString& big, const QString& small, const QColor& c) {
  return QStringLiteral("<div style='color:%1'><span style='font-size:22px; font-weight:bold'>%2</span><br>"
                        "<span style='font-size:14px'>%3</span></div>")
      .arg(c.name(), big.toHtmlEscaped(), small.toHtmlEscaped());
}

QString f2(double v) { return QString::number(v, 'f', 2); }
QString f3(double v) { return QString::number(v, 'f', 3); }

// Confidences of the MediaPipe models for the log: palm detector score, hand
// presence of the landmark model, its P(right hand); then crop size and sharpness.
QString detectText(const FrameResult& r) {
  const palm::HandLandmarks& h = r.hand;
  if (h.palmScore <= 0.f) return QStringLiteral("palm -");
  QString s = QStringLiteral("palm %1 pres %2").arg(f2(h.palmScore), f2(h.presence));
  if (!h.found) return s;
  s += QStringLiteral(" right %1 box %2px").arg(f2(h.handedness)).arg(r.check.box.width);
  if (r.check.state == palm::HandState::Moving)  // 1e9: no hand in the previous frame
    s += QStringLiteral(" motion %1").arg(r.check.motion < 1e6f ? f3(r.check.motion) : QStringLiteral("-"));
  if (r.check.sharp > 0) s += QStringLiteral(" sharp %1").arg(r.check.sharp, 0, 'f', 0);
  return s;
}

// Capture gate measurements of a frame, for calibrating the limits: "xoay" is
// the rotation in the image plane, "nghiêng" the palm plane's tilt out of it,
// "lệch" the tilt difference to the closest sample of this enrollment.
QString gateText(const FrameResult& r) {
  const palm::GateReport& g = r.gate;
  QString s = QStringLiteral("cỡ %1 sáng %2 sat %3 xoay %4° nghiêng %5° ngón %6")
                  .arg(f2(g.palmFrac), f2(g.mean), f3(g.satFrac), f2(g.angleDeg), QString::number(g.tiltDeg, 'f', 1),
                       f2(g.fingerOpen));
  if (g.poseSepDeg >= 0) s += QStringLiteral(" lệch %1°").arg(g.poseSepDeg, 0, 'f', 1);
  return s + QStringLiteral(" | %1 fps").arg(r.fps, 0, 'f', 1);
}

}  // namespace

MainWindow::MainWindow(const palm::Config& cfg, const std::string& source, bool mirrorSource, const QString& logPath)
    : cfg_(cfg), gallery_(cfg.modelVersion) {
  setWindowTitle(QStringLiteral("Palmprint - đăng ký tay phải"));
  log_ = new EventLog(logPath);  // first: everything below may log
  log_->setFixedHeight(170);
  log_->add("APP", QStringLiteral("palm_gui khởi động, gallery %1").arg(QString::fromStdString(cfg_.galleryPath)));
  if (log_->fileOpen())
    log_->add("APP", QStringLiteral("file log: %1").arg(log_->filePath()));
  else if (!logPath.isEmpty())
    log_->add("APP", QStringLiteral("không ghi được file log %1: %2").arg(logPath, log_->fileError()), EventLog::Warn);
  log_->add("APP", QStringLiteral("ngưỡng: nhận dạng cos >= %1 trong %2 khung liên tiếp | trùng khi đăng ký cos > %3 | "
                                  "%4 mẫu, cần %5 mẫu nhất quán | blur_min %6")
                       .arg(f2(cfg_.acceptCosMin))
                       .arg(cfg_.session.identifyConfirm)
                       .arg(f2(cfg_.dupCosMax))
                       .arg(cfg_.session.samples)
                       .arg(cfg_.tmpl.minValid)
                       .arg(cfg_.capture.blurMin, 0, 'f', 0));
  log_->add("APP", QStringLiteral("tư thế: nghiêng <= %1° (mẫu đầu <= %2°), các mẫu lệch nhau >= %3° | sáng %4..%5")
                       .arg(cfg_.gate.maxTiltDeg, 0, 'f', 0)
                       .arg(cfg_.gate.frontalMaxDeg, 0, 'f', 0)
                       .arg(cfg_.gate.minPoseSepDeg, 0, 'f', 0)
                       .arg(cfg_.gate.minMean, 0, 'f', 0)
                       .arg(cfg_.gate.maxMean, 0, 'f', 0));
  try {
    gallery_.load(cfg_.galleryPath);
    log_->add("APP", QStringLiteral("%1 người đã đăng ký, ID tự động kế tiếp %2")
                         .arg(gallery_.entries().size())
                         .arg(QString::fromStdString(gallery_.nextId())));
    // gallery files before v3 hold one mean template per person, from a single tilt
    const auto single = std::count_if(gallery_.entries().begin(), gallery_.entries().end(),
                                      [](const palm::GalleryEntry& e) { return e.samples.size() == 1; });
    if (single > 0)
      log_->add("APP", QStringLiteral("%1 người chỉ có 1 mẫu (gallery cũ, một góc nghiêng): nên xóa và đăng ký lại").arg(single),
                EventLog::Warn);
  } catch (const palm::GalleryError& e) {
    using Kind = palm::GalleryError::Kind;
    if (e.kind() == Kind::Corrupt || e.kind() == Kind::ModelMismatch) {
      const std::string bak = cfg_.galleryPath + ".bak";
      std::error_code ec;
      std::filesystem::rename(cfg_.galleryPath, bak, ec);
      log_->add("APP", QStringLiteral("không đọc được gallery (%1); đổi tên thành %2, bắt đầu gallery mới")
                           .arg(QString::fromStdString(e.what()), QString::fromStdString(bak)),
                EventLog::Fail);
      QMessageBox::warning(this, QStringLiteral("Gallery"),
                           QStringLiteral("Không đọc được gallery (%1).\nFile cũ được đổi tên thành %2; bắt đầu gallery mới.")
                               .arg(QString::fromStdString(e.what()), QString::fromStdString(bak)));
    } else {
      // locked, unreadable or written by a newer program: the file may be fine,
      // so it is left alone and nothing may be saved over it
      galleryOk_ = false;
      log_->add("APP", QStringLiteral("không mở được gallery (%1); không đăng ký / xóa được cho đến khi mở lại chương trình")
                           .arg(QString::fromStdString(e.what())),
                EventLog::Fail);
      QMessageBox::warning(this, QStringLiteral("Gallery"),
                           QStringLiteral("Không mở được gallery (%1).\nFile được giữ nguyên; đăng ký và xóa bị tắt. "
                                          "Đóng chương trình đang dùng file rồi mở lại.")
                               .arg(QString::fromStdString(e.what())));
    }
  }

  // ---- layout: video | panel
  video_ = new VideoView(cfg_.camera.displayMirror);
  video_->setPoseLimits(cfg_.gate);
  auto* panel = new QWidget;
  panel->setFixedWidth(340);
  auto* pv = new QVBoxLayout(panel);
  pv->setContentsMargins(6, 6, 6, 6);

  status_ = new QLabel(QStringLiteral("Đang khởi động..."));
  status_->setWordWrap(true);
  status_->setStyleSheet("color:#aaa; font-size:12px;");
  pv->addWidget(status_);

  auto* enrollBox = new QGroupBox(QStringLiteral("Đăng ký (chỉ tay PHẢI)"));
  auto* ev = new QVBoxLayout(enrollBox);
  auto* idRow = new QHBoxLayout;
  idRow->addWidget(new QLabel(QStringLiteral("ID:")));
  idEdit_ = new QLineEdit;
  idEdit_->setMaxLength(64);
  idRow->addWidget(idEdit_);
  ev->addLayout(idRow);
  auto* btnRow = new QHBoxLayout;
  enrollBtn_ = new QPushButton(QStringLiteral("Bắt đầu đăng ký"));
  cancelBtn_ = new QPushButton(QStringLiteral("Hủy"));
  btnRow->addWidget(enrollBtn_);
  btnRow->addWidget(cancelBtn_);
  ev->addLayout(btnRow);
  progress_ = new QProgressBar;
  progress_->setRange(0, cfg_.session.samples);
  progress_->setFormat(QStringLiteral("%v/%m mẫu"));
  ev->addWidget(progress_);
  auto* thumbRow = new QHBoxLayout;
  for (int i = 0; i < cfg_.session.samples; ++i) {
    thumbs_.push_back(imageBox(52));
    thumbRow->addWidget(thumbs_.back());
  }
  thumbRow->addStretch();
  ev->addLayout(thumbRow);
  enrollMsg_ = new QLabel;
  enrollMsg_->setWordWrap(true);
  ev->addWidget(enrollMsg_);
  pv->addWidget(enrollBox);

  auto* idBox = new QGroupBox(QStringLiteral("Nhận dạng 1:N"));
  auto* iv = new QVBoxLayout(idBox);
  identifyBtn_ = new QPushButton(QStringLiteral("Bắt đầu nhận dạng"));
  identifyBtn_->setCheckable(true);
  iv->addWidget(identifyBtn_);
  result_ = new QLabel;
  result_->setTextFormat(Qt::RichText);
  result_->setWordWrap(true);
  result_->setMinimumHeight(56);
  iv->addWidget(result_);
  auto* imgRow = new QHBoxLayout;
  cropView_ = imageBox(112);
  roiView_ = imageBox(112);
  cropView_->setToolTip(QStringLiteral("Ảnh bàn tay đưa vào model"));
  roiView_->setToolTip(QStringLiteral("Vùng lòng bàn tay model căn chỉnh (112x112)"));
  imgRow->addWidget(cropView_);
  imgRow->addWidget(roiView_);
  imgRow->addStretch();
  iv->addLayout(imgRow);
  pv->addWidget(idBox);

  auto* debugBox = new QGroupBox(QStringLiteral("Ảnh debug (vùng đo ánh sáng)"));
  auto* dv = new QHBoxLayout(debugBox);
  debugView_ = imageBox(112);
  debugView_->setToolTip(QStringLiteral("Vùng lòng bàn tay dùng để đo ánh sáng; pixel cháy sáng (>= 250) tô đỏ"));
  debugInfo_ = new QLabel;
  debugInfo_->setWordWrap(true);
  debugInfo_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
  dv->addWidget(debugView_);
  dv->addWidget(debugInfo_, 1);
  pv->addWidget(debugBox);

  auto* listBox = new QGroupBox(QStringLiteral("Đã đăng ký"));
  auto* lv = new QVBoxLayout(listBox);
  list_ = new QListWidget;
  lv->addWidget(list_);
  deleteBtn_ = new QPushButton(QStringLiteral("Xóa người đã chọn"));
  lv->addWidget(deleteBtn_);
  pv->addWidget(listBox, 1);

  auto* bottomRow = new QHBoxLayout;
  logBtn_ = new QPushButton(QStringLiteral("Log (F2)"));
  logBtn_->setCheckable(true);
  logBtn_->setChecked(true);
  logBtn_->setStyleSheet("QPushButton { background:#444; } QPushButton:checked { background:#3d5a80; }");
  auto* quit = new QPushButton(QStringLiteral("Thoát (Esc)"));
  bottomRow->addWidget(logBtn_);
  bottomRow->addWidget(quit);
  pv->addLayout(bottomRow);

  auto* left = new QWidget;  // video above the log
  auto* lv2 = new QVBoxLayout(left);
  lv2->setContentsMargins(0, 0, 0, 0);
  lv2->setSpacing(0);
  lv2->addWidget(video_, 1);
  lv2->addWidget(log_);
  auto* central = new QWidget;
  auto* h = new QHBoxLayout(central);
  h->setContentsMargins(0, 0, 0, 0);
  h->setSpacing(0);
  h->addWidget(left, 1);
  h->addWidget(panel);
  setCentralWidget(central);
  setStyleSheet("QMainWindow, QWidget { background:#2b2b2b; color:#eee; font-size:14px; }"
                "QGroupBox { border:1px solid #555; border-radius:4px; margin-top:14px; font-weight:bold; }"
                "QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }"
                "QPushButton { background:#3d5a80; border:none; padding:7px; border-radius:4px; }"
                "QPushButton:checked { background:#c0392b; }"
                "QPushButton:disabled { background:#444; color:#777; }"
                "QLineEdit, QListWidget { background:#1e1e1e; border:1px solid #555; padding:4px; }"
                "QProgressBar { border:1px solid #555; border-radius:3px; text-align:center; }"
                "QProgressBar::chunk { background:#27ae60; }");

  connect(enrollBtn_, &QPushButton::clicked, this, &MainWindow::startEnroll);
  connect(cancelBtn_, &QPushButton::clicked, this, &MainWindow::cancelEnroll);
  connect(identifyBtn_, &QPushButton::toggled, this, &MainWindow::toggleIdentify);
  connect(deleteBtn_, &QPushButton::clicked, this, &MainWindow::deleteSelected);
  connect(quit, &QPushButton::clicked, this, &QWidget::close);
  connect(logBtn_, &QPushButton::toggled, log_, &QWidget::setVisible);

  refreshList();
  idEdit_->setText(nextFreeId());

  // ---- worker thread
  worker_ = new CameraWorker(cfg_, source, mirrorSource);
  worker_->moveToThread(&thread_);
  connect(&thread_, &QThread::started, worker_, &CameraWorker::run);
  connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);
  connect(worker_, &CameraWorker::frameReady, this, &MainWindow::onFrame);
  connect(worker_, &CameraWorker::status, this, &MainWindow::onWorkerStatus);
  connect(worker_, &CameraWorker::failed, this, &MainWindow::onWorkerFailed);
  thread_.start();
  setMode(Mode::Idle);
}

MainWindow::~MainWindow() {
  worker_->stop();
  thread_.quit();
  thread_.wait();
}

void MainWindow::keyPressEvent(QKeyEvent* e) {
  if (e->key() == Qt::Key_Escape) close();
  else if (e->key() == Qt::Key_F11) isFullScreen() ? showNormal() : showFullScreen();
  else if (e->key() == Qt::Key_F2) logBtn_->toggle();
  else QMainWindow::keyPressEvent(e);
}

void MainWindow::onWorkerStatus(const QString& text) {
  status_->setText(text);
  video_->setMessage(text);
  log_->add("CAM", text);
}

void MainWindow::onWorkerFailed(const QString& text) {
  status_->setText(QStringLiteral("<span style='color:#e74c3c'>%1</span>").arg(text.toHtmlEscaped()));
  video_->setMessage(text);
  log_->add("CAM", text, EventLog::Fail);
}

void MainWindow::setMode(Mode m) {
  mode_ = m;
  worker_->setEmbedInterval(m == Mode::Enroll ? cfg_.session.gapMs : m == Mode::Identify ? 0 : -1);
  worker_->setGated(m == Mode::Enroll);
  const bool idle = m == Mode::Idle;
  enrollBtn_->setEnabled(idle);
  idEdit_->setEnabled(idle);
  cancelBtn_->setEnabled(m == Mode::Enroll);
  identifyBtn_->setEnabled(m != Mode::Enroll);
  deleteBtn_->setEnabled(idle);
  {
    const QSignalBlocker block(identifyBtn_);
    identifyBtn_->setChecked(m == Mode::Identify);
    identifyBtn_->setText(m == Mode::Identify ? QStringLiteral("Dừng nhận dạng") : QStringLiteral("Bắt đầu nhận dạng"));
  }
  if (idle) video_->setBanner({}, {});
}

void MainWindow::startEnroll() {
  const QString id = idEdit_->text().trimmed();
  if (!galleryOk_) {
    enrollMsg_->setText(QStringLiteral("Gallery không mở được: không thể đăng ký."));
    enrollMsg_->setStyleSheet("color:#e74c3c;");
    return;
  }
  if (id.isEmpty()) {
    enrollMsg_->setText(QStringLiteral("Nhập ID trước khi đăng ký."));
    return;
  }
  const auto& es = gallery_.entries();
  const bool exists = std::any_of(es.begin(), es.end(), [&](const palm::GalleryEntry& e) { return e.id == id.toStdString(); });
  if (exists && QMessageBox::question(this, QStringLiteral("ID đã tồn tại"),
                                      QStringLiteral("'%1' đã đăng ký tay phải. Mỗi người chỉ có một mẫu.\n"
                                                     "Đăng ký lại sẽ thay mẫu cũ. Tiếp tục?").arg(id)) != QMessageBox::Yes)
    return;
  enrollId_ = id;
  enrollFeats_.clear();
  for (auto* t : thumbs_) t->clear();
  progress_->setValue(0);
  enrollMsg_->setText(QStringLiteral("Giơ tay PHẢI, lòng bàn tay thẳng hướng vào camera, xòe các ngón."));
  enrollMsg_->setStyleSheet("");
  enrollClock_.start();
  setMode(Mode::Enroll);
  log_->add("ENROLL", QStringLiteral("bắt đầu đăng ký %1%2: %3 mẫu, cách nhau %4 ms, tối đa %5 s")
                          .arg(id, exists ? QStringLiteral(" (thay mẫu cũ)") : QString())
                          .arg(cfg_.session.samples)
                          .arg(cfg_.session.gapMs)
                          .arg(cfg_.session.timeoutS));
}

void MainWindow::beginEnroll(const QString& id) {
  idEdit_->setText(id);
  startEnroll();
}

void MainWindow::beginIdentify() { identifyBtn_->setChecked(true); }

void MainWindow::cancelEnroll() {
  enrollMsg_->setText(QStringLiteral("Đã hủy đăng ký."));
  setMode(Mode::Idle);
  log_->add("ENROLL", QStringLiteral("hủy đăng ký %1 sau %2/%3 mẫu").arg(enrollId_).arg(enrollFeats_.size()).arg(cfg_.session.samples),
            EventLog::Warn);
}

void MainWindow::toggleIdentify(bool on) {
  lastId_.clear();
  streak_ = 0;
  result_->clear();
  lastEmbedClock_.start();
  setMode(on ? Mode::Identify : Mode::Idle);
  if (!on)
    log_->add("ID", QStringLiteral("dừng nhận dạng"));
  else if (gallery_.entries().empty())
    log_->add("ID", QStringLiteral("bắt đầu nhận dạng, nhưng chưa có ai đăng ký"), EventLog::Warn);
  else
    log_->add("ID", QStringLiteral("bắt đầu nhận dạng: %1 người, cos >= %2 trong %3 khung liên tiếp")
                        .arg(gallery_.entries().size())
                        .arg(f2(cfg_.acceptCosMin))
                        .arg(cfg_.session.identifyConfirm));
}

void MainWindow::logFrame(const FrameResult& r) {
  // gate status changes, at most one line per 0.5 s (a hand at a limit flickers)
  const int st = static_cast<int>(r.gate.status);
  if (st != loggedState_ && (!handLogClock_.isValid() || handLogClock_.elapsed() >= 500)) {
    loggedState_ = st;
    handLogClock_.start();
    const QString msg = QString::fromUtf8(palm::gateMessage(r.gate.status));
    const QString measured = gateText(r) + " | " + detectText(r);
    log_->add("HAND", QStringLiteral("%1 | %2 | detect %3 ms").arg(msg, measured).arg(r.msDetect, 0, 'f', 0));
  }
  // timing summary every 30 s
  if (!perfClock_.isValid()) perfClock_.start();
  ++perfFrames_;
  perfDetectMs_ += r.msDetect;
  if (r.embedded) ++perfEmbeds_, perfEmbedMs_ += r.msEmbed;
  if (perfClock_.elapsed() >= 30000) {
    log_->add("PERF", QStringLiteral("%1 fps | phát hiện tay %2 ms/khung | embedding %3 lần, %4 ms/lần")
                          .arg(r.fps, 0, 'f', 1)
                          .arg(perfDetectMs_ / perfFrames_, 0, 'f', 1)
                          .arg(perfEmbeds_)
                          .arg(perfEmbeds_ ? perfEmbedMs_ / perfEmbeds_ : 0.0, 0, 'f', 1));
    perfFrames_ = perfEmbeds_ = 0;
    perfDetectMs_ = perfEmbedMs_ = 0;
    perfClock_.restart();
  }
}

void MainWindow::onFrame(const FrameResult& r) {
  video_->setFrame(r);
  logFrame(r);
  status_->setText(QStringLiteral("%1 fps | tay %2 ms%3 | %4 người đã đăng ký")
                       .arg(r.fps, 0, 'f', 1)
                       .arg(r.msDetect, 0, 'f', 0)
                       .arg(r.embedded ? QStringLiteral(" | embedding %1 ms").arg(r.msEmbed, 0, 'f', 0) : QString())
                       .arg(gallery_.entries().size()));
  if (!r.lightDebug.isNull()) {
    debugView_->setPixmap(pix(r.lightDebug, 112));
    debugInfo_->setText(QStringLiteral("sáng TB %1 (%2..%3)\ncháy %4% (tối đa %5%)")
                            .arg(r.gate.mean, 0, 'f', 0)
                            .arg(cfg_.gate.minMean, 0, 'f', 0)
                            .arg(cfg_.gate.maxMean, 0, 'f', 0)
                            .arg(r.gate.satFrac * 100, 0, 'f', 1)
                            .arg(cfg_.gate.maxSatFrac * 100, 0, 'f', 0));
  } else {
    debugView_->clear();
    debugInfo_->clear();
  }
  if (r.embedded) {
    cropView_->setPixmap(pix(r.crop, 112));
    roiView_->setPixmap(pix(r.roi, 112));
  }
  if (mode_ == Mode::Enroll) {
    if (r.embedded) addEnrollSample(r);
    if (mode_ == Mode::Enroll) {
      if (enrollClock_.elapsed() > cfg_.session.timeoutS * 1000LL) {
        enrollMsg_->setText(QStringLiteral("Hết thời gian (%1 s). Thử lại.").arg(cfg_.session.timeoutS));
        setMode(Mode::Idle);
        log_->add("ENROLL", QStringLiteral("THẤT BẠI %1: hết thời gian %2 s, mới có %3/%4 mẫu")
                                .arg(enrollId_)
                                .arg(cfg_.session.timeoutS)
                                .arg(enrollFeats_.size())
                                .arg(cfg_.session.samples),
                  EventLog::Fail);
      } else {
        video_->setBanner(QStringLiteral("Đăng ký %1: %2/%3 mẫu").arg(enrollId_).arg(enrollFeats_.size()).arg(cfg_.session.samples),
                          kBlue);
      }
    }
  } else if (mode_ == Mode::Identify) {
    if (r.embedded) {
      identify(r);
    } else if (lastEmbedClock_.elapsed() > 1500) {
      streak_ = 0;
      lastId_.clear();
      result_->setText(resultHtml(QStringLiteral("—"), QStringLiteral("Chưa có tay phải hợp lệ"), QColor(150, 150, 150)));
      video_->setBanner({}, {});
    }
  }
}

void MainWindow::addEnrollSample(const FrameResult& r) {
  const size_t k = enrollFeats_.size();
  // EE-PRnet gives no probability: its confidence is the cosine to other samples / users
  QString line = QStringLiteral("mẫu %1/%2").arg(k + 1).arg(cfg_.session.samples);
  if (k >= 2) line += QStringLiteral(" | cos với mẫu 1 %1").arg(f3(palm::dot(r.feat, enrollFeats_.front())));
  if (k >= 1) line += QStringLiteral(" | cos với mẫu trước %1").arg(f3(palm::dot(r.feat, enrollFeats_.back())));
  const auto other = gallery_.search(r.feat, 1, enrollId_.toStdString());
  if (!other.empty())
    line += QStringLiteral(" | gần nhất người khác %1 %2").arg(QString::fromStdString(other[0].id), f3(other[0].score));
  line += QStringLiteral(" | %1 | %2 | embed %3 ms").arg(gateText(r), detectText(r)).arg(r.msEmbed, 0, 'f', 0);
  log_->add("ENROLL", line);
  enrollFeats_.push_back(r.feat);
  if (k < thumbs_.size()) thumbs_[k]->setPixmap(pix(r.roi, 52));
  progress_->setValue(static_cast<int>(enrollFeats_.size()));
  if (static_cast<int>(enrollFeats_.size()) >= cfg_.session.samples) {
    finishEnroll();
    return;
  }
  enrollMsg_->setText(QStringLiteral("Đã chụp %1/%2. Nghiêng nhẹ lòng bàn tay sang hướng khác (chấm trắng ra xa "
                                     "các chấm đã chụp trên bản đồ góc), rồi giữ yên.")
                          .arg(enrollFeats_.size()).arg(cfg_.session.samples));
}

void MainWindow::finishEnroll() {
  setMode(Mode::Idle);
  const palm::TemplateResult t = palm::buildTemplate(enrollFeats_, cfg_.tmpl);
  QString cos;
  for (float c : t.cosToMedoid) cos += QString::number(c, 'f', 2) + " ";
  cos = cos.trimmed();
  if (!t.ok) {
    enrollMsg_->setText(QStringLiteral("THẤT BẠI: chỉ %1/%2 mẫu nhất quán (cos với mẫu trung tâm: %3). Đăng ký lại.")
                            .arg(t.kept.size()).arg(enrollFeats_.size()).arg(cos));
    enrollMsg_->setStyleSheet("color:#e74c3c;");
    log_->add("ENROLL", QStringLiteral("THẤT BẠI %1: %2/%3 mẫu nhất quán, cần %4 (cos với mẫu trung tâm: %5; loại dưới %6)")
                            .arg(enrollId_)
                            .arg(t.kept.size())
                            .arg(enrollFeats_.size())
                            .arg(cfg_.tmpl.minValid)
                            .arg(cos, f2(cfg_.tmpl.outlierCosMin)),
              EventLog::Fail);
    return;
  }
  const std::string id = enrollId_.toStdString();
  const std::vector<palm::Vec> samples = palm::keptSamples(enrollFeats_, t);
  const auto dup = gallery_.search(samples, 1, id);  // closest pair of samples
  if (!dup.empty() && dup[0].score > cfg_.dupCosMax) {
    enrollMsg_->setText(QStringLiteral("TỪ CHỐI: bàn tay này đã đăng ký với ID '%1' (cos %2). Mỗi người chỉ đăng ký một lần.")
                            .arg(QString::fromStdString(dup[0].id)).arg(dup[0].score, 0, 'f', 3));
    enrollMsg_->setStyleSheet("color:#e74c3c;");
    log_->add("ENROLL", QStringLiteral("TỪ CHỐI %1: trùng người đã đăng ký %2 (cos %3 > %4)")
                            .arg(enrollId_, QString::fromStdString(dup[0].id), f3(dup[0].score), f2(cfg_.dupCosMax)),
              EventLog::Fail);
    return;
  }
  const bool replaced = gallery_.contains(id);
  gallery_.upsert({id, samples});
  try {
    gallery_.save(cfg_.galleryPath);
  } catch (const std::exception& e) {
    enrollMsg_->setText(QStringLiteral("Lỗi lưu gallery: %1").arg(e.what()));
    enrollMsg_->setStyleSheet("color:#e74c3c;");
    log_->add("ENROLL", QStringLiteral("THẤT BẠI %1: không lưu được gallery: %2").arg(enrollId_, e.what()), EventLog::Fail);
    return;
  }
  log_->add("ENROLL", QStringLiteral("THÀNH CÔNG %1%2: %3/%4 mẫu, cos với mẫu trung tâm %5 | %6")
                          .arg(enrollId_, replaced ? QStringLiteral(" (thay mẫu cũ)") : QString())
                          .arg(t.kept.size())
                          .arg(enrollFeats_.size())
                          .arg(cos,
                               dup.empty() ? QStringLiteral("chưa có người khác")
                                           : QStringLiteral("gần nhất người khác %1 cos %2 (ngưỡng trùng %3)")
                                                 .arg(QString::fromStdString(dup[0].id), f3(dup[0].score),
                                                      f2(cfg_.dupCosMax))),
            EventLog::Ok);
  enrollMsg_->setText(QStringLiteral("THÀNH CÔNG: đã đăng ký tay phải của '%1' (%2/%3 mẫu).")
                          .arg(enrollId_).arg(t.kept.size()).arg(enrollFeats_.size()));
  enrollMsg_->setStyleSheet("color:#2ecc71;");
  refreshList();
  idEdit_->setText(nextFreeId());
}

void MainWindow::identify(const FrameResult& r) {
  lastEmbedClock_.restart();
  if (gallery_.entries().empty()) {
    result_->setText(resultHtml(QStringLiteral("Chưa có ai đăng ký"), {}, kAmber));
    return;
  }
  const auto m = gallery_.search(r.feat, 2);
  const QString id = QString::fromStdString(m[0].id);
  const bool accept = m[0].score >= cfg_.acceptCosMin;
  if (accept && id == lastId_) {
    ++streak_;
  } else {
    streak_ = accept ? 1 : 0;
    lastId_ = accept ? id : QString();
  }
  const int confirm = cfg_.session.identifyConfirm;
  // EE-PRnet gives no probability: its confidence is the cosine to the best
  // match, and the gap to the second best
  QString line = QStringLiteral("%1 cos %2").arg(id, f3(m[0].score));
  if (m.size() > 1)
    line += QStringLiteral(" | thứ 2 %1 cos %2 (cách %3)")
                .arg(QString::fromStdString(m[1].id), f3(m[1].score), f3(m[0].score - m[1].score));
  line += accept ? QStringLiteral(" | ĐẠT %1/%2").arg(std::min(streak_, confirm)).arg(confirm)
                 : QStringLiteral(" | DƯỚI NGƯỠNG %1").arg(f2(cfg_.acceptCosMin));
  line += QStringLiteral(" | %1 | embed %2 ms").arg(detectText(r)).arg(r.msEmbed, 0, 'f', 0);
  log_->add("ID", line, accept ? EventLog::Info : EventLog::Warn);
  if (accept && streak_ == confirm)
    log_->add("ID", QStringLiteral("XÁC NHẬN: xin chào %1 (cos %2, %3 khung liên tiếp)").arg(id, f3(m[0].score)).arg(confirm),
              EventLog::Ok);
  const QString detail = QStringLiteral("cos %1 (ngưỡng %2)").arg(m[0].score, 0, 'f', 3).arg(cfg_.acceptCosMin, 0, 'f', 2);
  if (accept && streak_ >= cfg_.session.identifyConfirm) {
    result_->setText(resultHtml(id, detail, kGreen));
    video_->setBanner(QStringLiteral("Xin chào %1").arg(id), kGreen);
  } else if (accept) {
    result_->setText(resultHtml(QStringLiteral("%1 ?").arg(id), QStringLiteral("đang xác nhận %1/%2 - %3")
                                                                  .arg(streak_).arg(cfg_.session.identifyConfirm).arg(detail),
                                kAmber));
  } else {
    result_->setText(resultHtml(QStringLiteral("Không nhận ra"), QStringLiteral("gần nhất: %1, %2").arg(id, detail), kRed));
    video_->setBanner(QStringLiteral("Không nhận ra"), kRed);
  }
}

void MainWindow::refreshList() {
  list_->clear();
  for (const auto& e : gallery_.entries()) {
    auto* item = new QListWidgetItem(QStringLiteral("%1   (%2 mẫu)").arg(QString::fromStdString(e.id)).arg(e.samples.size()));
    item->setData(Qt::UserRole, QString::fromStdString(e.id));
    list_->addItem(item);
  }
}

QString MainWindow::nextFreeId() const { return QString::fromStdString(gallery_.nextId()); }

void MainWindow::deleteSelected() {
  const QListWidgetItem* item = list_->currentItem();
  if (!item) return;
  const QString id = item->data(Qt::UserRole).toString();
  if (!galleryOk_) {
    QMessageBox::warning(this, QStringLiteral("Lỗi"), QStringLiteral("Gallery không mở được: không thể xóa."));
    return;
  }
  if (QMessageBox::question(this, QStringLiteral("Xóa"), QStringLiteral("Xóa '%1' khỏi gallery?").arg(id)) != QMessageBox::Yes)
    return;
  gallery_.remove(id.toStdString());
  try {
    gallery_.save(cfg_.galleryPath);
    log_->add("APP", QStringLiteral("đã xóa %1, còn %2 người").arg(id).arg(gallery_.entries().size()));
  } catch (const std::exception& e) {
    log_->add("APP", QStringLiteral("xóa %1: không lưu được gallery: %2").arg(id, e.what()), EventLog::Fail);
    QMessageBox::warning(this, QStringLiteral("Lỗi"), QString::fromStdString(e.what()));
  }
  refreshList();
  idEdit_->setText(nextFreeId());
}
