#include "camera_worker.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "palm_model.h"

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

QImage toQImage(const cv::Mat& bgr) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
}

QImage rgbFloatToQImage(const cv::Mat& rgb32f) {
  cv::Mat u8;
  rgb32f.convertTo(u8, CV_8UC3);  // saturating
  return QImage(u8.data, u8.cols, u8.rows, static_cast<int>(u8.step), QImage::Format_RGB888).copy();
}

// Reads the camera on its own thread and keeps only the newest frame, so a
// slow processing step never works on stale frames.
class Grabber {
 public:
  explicit Grabber(cv::VideoCapture& cap) : cap_(cap), thread_([this] { loop(); }) {}
  ~Grabber() {
    stop_ = true;
    thread_.join();
  }
  bool dead() const { return dead_; }

  // waits up to timeoutMs for a frame newer than `seen`
  bool next(cv::Mat& out, long& seen, int timeoutMs) {
    std::unique_lock<std::mutex> lk(m_);
    if (!cv_.wait_for(lk, std::chrono::milliseconds(timeoutMs), [&] { return seq_ > seen || dead_; }) || dead_)
      return false;
    latest_.copyTo(out);
    seen = seq_;
    return true;
  }

 private:
  void loop() {
    cv::Mat f;
    int failures = 0;
    while (!stop_) {
      if (!cap_.read(f) || f.empty()) {
        if (++failures > 100) {
          dead_ = true;
          cv_.notify_all();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }
      failures = 0;
      {
        std::lock_guard<std::mutex> lk(m_);
        std::swap(latest_, f);
        ++seq_;
      }
      cv_.notify_all();
    }
  }

  cv::VideoCapture& cap_;
  std::mutex m_;
  std::condition_variable cv_;
  cv::Mat latest_;
  long seq_ = 0;
  std::atomic<bool> stop_{false}, dead_{false};
  std::thread thread_;  // last: starts after the members above exist
};

}  // namespace

CameraWorker::CameraWorker(const palm::Config& cfg, std::string source, bool mirrorSource)
    : cfg_(cfg), source_(std::move(source)), mirrorSource_(mirrorSource) {}

CameraWorker::~CameraWorker() = default;

void CameraWorker::run() {
  try {
    emit status(QStringLiteral("Đang nạp model..."));
    palm::PalmModel model(cfg_.model);
    palm::HandDetector det(cfg_.hand.palmOnnx, cfg_.hand.landmarkOnnx, 2);

    cv::Mat still;
    cv::VideoCapture cap;
    if (!source_.empty()) {
      still = cv::imread(source_, cv::IMREAD_COLOR);
      if (still.empty()) cap.open(source_);
    } else {
      cap.open(cfg_.camera.device, cv::CAP_V4L2);
      if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg_.camera.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg_.camera.height);
        cap.set(cv::CAP_PROP_FPS, 30);
      }
    }
    if (still.empty() && !cap.isOpened()) {
      emit failed(source_.empty() ? QStringLiteral("Không mở được camera /dev/video%1").arg(cfg_.camera.device)
                                  : QStringLiteral("Không mở được nguồn %1").arg(QString::fromStdString(source_)));
      return;
    }
    if (!still.empty() && mirrorSource_) cv::flip(still, still, 1);
    if (still.empty())
      emit status(QStringLiteral("Camera %1x%2").arg(cap.get(cv::CAP_PROP_FRAME_WIDTH)).arg(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    else
      emit status(QStringLiteral("Ảnh tĩnh %1x%2").arg(still.cols).arg(still.rows));

    std::unique_ptr<Grabber> grabber;
    if (still.empty()) grabber = std::make_unique<Grabber>(cap);
    palm::HandLandmarks prev;
    palm::CaptureGate gate(cfg_.gate);
    bool wasGated = false;
    auto lastEmbed = Clock::now() - std::chrono::hours(1);
    auto tPrev = Clock::now(), tLog = Clock::now();
    double fps = 0, sumDetect = 0, sumEmbed = 0;
    int nFrames = 0, nEmbed = 0;
    long seen = 0;
    auto nextStill = Clock::now();
    cv::Mat frame;
    while (!stop_) {
      if (grabber) {
        if (!grabber->next(frame, seen, 500)) {
          if (grabber->dead()) {
            emit failed(QStringLiteral("Mất tín hiệu camera"));
            return;
          }
          continue;
        }
        if (mirrorSource_ && !source_.empty()) cv::flip(frame, frame, 1);
      } else {
        // replay the still at most at 30 fps, like a camera (no sleep when processing is slower)
        std::this_thread::sleep_until(nextStill);
        nextStill = std::max(nextStill, Clock::now()) + std::chrono::milliseconds(33);
        frame = still.clone();
      }

      FrameResult r;
      const auto t0 = Clock::now();
      r.hand = det.detect(frame);
      r.check = palm::checkHand(frame, r.hand, prev, cfg_.capture);
      prev = r.hand;
      r.msDetect = msSince(t0);

      // the gate runs on every frame, so its stability window is full when enrollment starts
      const double nowMs = std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
      r.gated = gated_;
      if (r.gated != wasGated) gate.setCollecting(r.gated);  // a new enrollment starts with no pose taken
      wasGated = r.gated;
      r.gate = gate.update(frame, r.check, r.hand, nowMs);

      const int every = embedIntervalMs_;
      const bool ready = r.gated ? r.gate.status == palm::GateStatus::Ready : r.gate.framePasses();
      if (every >= 0 && ready && msSince(lastEmbed) >= every) {
        const auto t1 = Clock::now();
        const cv::Mat crop = frame(r.check.box);
        const palm::Embedding e = model.run(crop, false);  // camera frames are not mirrored
        r.feat = palm::l2normalize(e.feature);
        r.roi = rgbFloatToQImage(e.roi);
        cv::Mat thumb;
        cv::resize(crop, thumb, {112, 112}, 0, 0, cv::INTER_AREA);
        r.crop = toQImage(thumb);
        r.embedded = true;
        r.msEmbed = msSince(t1);
        lastEmbed = Clock::now();
        if (r.gated) {
          gate.addTaken(r.gate.normal);  // the next sample needs another tilt
          gate.restart();                // and its own stable window
        }
      }
      if (r.gated) r.taken = gate.taken();

      r.viewScale = std::min(1.0, 1280.0 / frame.cols);
      cv::Mat view = frame;
      if (r.viewScale < 1) cv::resize(frame, view, {}, r.viewScale, r.viewScale, cv::INTER_AREA);
      r.view = toQImage(view);
      const double dt = msSince(tPrev);
      tPrev = Clock::now();
      fps = fps == 0 ? 1000.0 / dt : 0.9 * fps + 0.1 * 1000.0 / dt;
      r.fps = fps;
      if (verbose_) {
        ++nFrames, sumDetect += r.msDetect;
        if (r.embedded) ++nEmbed, sumEmbed += r.msEmbed;
        if (msSince(tLog) >= 1000) {
          std::fprintf(stderr, "[palm_gui] %.1f fps  hand %.1f ms  embed %.1f ms (%d)  state: %s\n", fps,
                       sumDetect / nFrames, nEmbed ? sumEmbed / nEmbed : 0.0, nEmbed, palm::stateMessage(r.check.state));
          tLog = Clock::now();
          nFrames = nEmbed = 0;
          sumDetect = sumEmbed = 0;
        }
      }
      emit frameReady(r);
    }
  } catch (const std::exception& e) {
    emit failed(QString::fromStdString(e.what()));
  }
}
