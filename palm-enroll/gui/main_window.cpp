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

}  // namespace

MainWindow::MainWindow(const palm::Config& cfg, const std::string& source, bool mirrorSource)
    : cfg_(cfg), gallery_(cfg.modelVersion) {
  setWindowTitle(QStringLiteral("Palmprint - đăng ký tay phải"));
  try {
    gallery_.load(cfg_.galleryPath);
  } catch (const std::exception& e) {
    const std::string bak = cfg_.galleryPath + ".bak";
    std::error_code ec;
    std::filesystem::rename(cfg_.galleryPath, bak, ec);
    QMessageBox::warning(this, QStringLiteral("Gallery"),
                         QStringLiteral("Không đọc được gallery (%1).\nFile cũ được đổi tên thành %2; bắt đầu gallery mới.")
                             .arg(QString::fromStdString(e.what()), QString::fromStdString(bak)));
  }

  // ---- layout: video | panel
  video_ = new VideoView(cfg_.camera.displayMirror);
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

  auto* listBox = new QGroupBox(QStringLiteral("Đã đăng ký"));
  auto* lv = new QVBoxLayout(listBox);
  list_ = new QListWidget;
  lv->addWidget(list_);
  deleteBtn_ = new QPushButton(QStringLiteral("Xóa người đã chọn"));
  lv->addWidget(deleteBtn_);
  pv->addWidget(listBox, 1);

  auto* quit = new QPushButton(QStringLiteral("Thoát (Esc)"));
  pv->addWidget(quit);

  auto* central = new QWidget;
  auto* h = new QHBoxLayout(central);
  h->setContentsMargins(0, 0, 0, 0);
  h->setSpacing(0);
  h->addWidget(video_, 1);
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
  else QMainWindow::keyPressEvent(e);
}

void MainWindow::onWorkerStatus(const QString& text) {
  status_->setText(text);
  video_->setMessage(text);
}

void MainWindow::onWorkerFailed(const QString& text) {
  status_->setText(QStringLiteral("<span style='color:#e74c3c'>%1</span>").arg(text.toHtmlEscaped()));
  video_->setMessage(text);
}

void MainWindow::setMode(Mode m) {
  mode_ = m;
  worker_->setEmbedInterval(m == Mode::Enroll ? cfg_.session.gapMs : m == Mode::Identify ? 0 : -1);
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
  enrollMsg_->setText(QStringLiteral("Giơ tay PHẢI, lòng bàn tay hướng vào camera, xòe các ngón."));
  enrollMsg_->setStyleSheet("");
  enrollClock_.start();
  setMode(Mode::Enroll);
}

void MainWindow::beginEnroll(const QString& id) {
  idEdit_->setText(id);
  startEnroll();
}

void MainWindow::beginIdentify() { identifyBtn_->setChecked(true); }

void MainWindow::cancelEnroll() {
  enrollMsg_->setText(QStringLiteral("Đã hủy đăng ký."));
  setMode(Mode::Idle);
}

void MainWindow::toggleIdentify(bool on) {
  lastId_.clear();
  streak_ = 0;
  result_->clear();
  lastEmbedClock_.start();
  setMode(on ? Mode::Identify : Mode::Idle);
}

void MainWindow::onFrame(const FrameResult& r) {
  video_->setFrame(r);
  status_->setText(QStringLiteral("%1 fps | tay %2 ms%3 | %4 người đã đăng ký")
                       .arg(r.fps, 0, 'f', 1)
                       .arg(r.msDetect, 0, 'f', 0)
                       .arg(r.embedded ? QStringLiteral(" | embedding %1 ms").arg(r.msEmbed, 0, 'f', 0) : QString())
                       .arg(gallery_.entries().size()));
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
  enrollFeats_.push_back(r.feat);
  if (k < thumbs_.size()) thumbs_[k]->setPixmap(pix(r.roi, 52));
  progress_->setValue(static_cast<int>(enrollFeats_.size()));
  if (static_cast<int>(enrollFeats_.size()) >= cfg_.session.samples) {
    finishEnroll();
    return;
  }
  enrollMsg_->setText(QStringLiteral("Đã chụp %1/%2. Xoay hoặc nghiêng nhẹ tay, rồi giữ yên.")
                          .arg(enrollFeats_.size()).arg(cfg_.session.samples));
}

void MainWindow::finishEnroll() {
  setMode(Mode::Idle);
  const palm::TemplateResult t = palm::buildTemplate(enrollFeats_, cfg_.tmpl);
  if (!t.ok) {
    QString cos;
    for (float c : t.cosToMedoid) cos += QString::number(c, 'f', 2) + " ";
    enrollMsg_->setText(QStringLiteral("THẤT BẠI: chỉ %1/%2 mẫu nhất quán (cos với mẫu trung tâm: %3). Đăng ký lại.")
                            .arg(t.kept.size()).arg(enrollFeats_.size()).arg(cos.trimmed()));
    enrollMsg_->setStyleSheet("color:#e74c3c;");
    return;
  }
  const std::string id = enrollId_.toStdString();
  const auto dup = gallery_.search(t.tmpl, 1, id);
  if (!dup.empty() && dup[0].score > cfg_.dupCosMax) {
    enrollMsg_->setText(QStringLiteral("TỪ CHỐI: bàn tay này đã đăng ký với ID '%1' (cos %2). Mỗi người chỉ đăng ký một lần.")
                            .arg(QString::fromStdString(dup[0].id)).arg(dup[0].score, 0, 'f', 3));
    enrollMsg_->setStyleSheet("color:#e74c3c;");
    return;
  }
  gallery_.upsert({id, static_cast<int>(t.kept.size()), t.tmpl});
  try {
    gallery_.save(cfg_.galleryPath);
  } catch (const std::exception& e) {
    enrollMsg_->setText(QStringLiteral("Lỗi lưu gallery: %1").arg(e.what()));
    enrollMsg_->setStyleSheet("color:#e74c3c;");
    return;
  }
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
    auto* item = new QListWidgetItem(QStringLiteral("%1   (%2 mẫu)").arg(QString::fromStdString(e.id)).arg(e.numSamples));
    item->setData(Qt::UserRole, QString::fromStdString(e.id));
    list_->addItem(item);
  }
}

QString MainWindow::nextFreeId() const {
  for (int i = 1;; ++i) {
    const QString id = QStringLiteral("user_%1").arg(i, 2, 10, QLatin1Char('0'));
    const auto& es = gallery_.entries();
    if (std::none_of(es.begin(), es.end(), [&](const palm::GalleryEntry& e) { return e.id == id.toStdString(); }))
      return id;
  }
}

void MainWindow::deleteSelected() {
  const QListWidgetItem* item = list_->currentItem();
  if (!item) return;
  const QString id = item->data(Qt::UserRole).toString();
  if (QMessageBox::question(this, QStringLiteral("Xóa"), QStringLiteral("Xóa '%1' khỏi gallery?").arg(id)) != QMessageBox::Yes)
    return;
  gallery_.remove(id.toStdString());
  try {
    gallery_.save(cfg_.galleryPath);
  } catch (const std::exception& e) {
    QMessageBox::warning(this, QStringLiteral("Lỗi"), QString::fromStdString(e.what()));
  }
  refreshList();
  idEdit_->setText(nextFreeId());
}
