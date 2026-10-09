#include "hand_detector.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

namespace palm {
namespace {

constexpr int kPalmIn = 192, kHandIn = 224;
constexpr float kScoreThr = 0.5f, kNmsThr = 0.3f, kPresenceThr = 0.8f;
const char* kIn[] = {"input_1"};
const char* kPalmOut[] = {"Identity", "Identity_1"};                            // [1,2016,18], [1,2016,1]
const char* kHandOut[] = {"Identity", "Identity_1", "Identity_2", "Identity_3"};  // landmarks, presence, handedness, world

struct Palm {
  cv::Point2d p0, p1;               // box corners
  std::array<cv::Point2d, 7> kp;    // wrist, 5 finger bases..., see mp_palmdet.py
  double score = 0;
};

// SSD anchors of the MediaPipe palm detector (identical to the table in mp_palmdet.py)
std::vector<cv::Point2d> makeAnchors() {
  std::vector<cv::Point2d> a;
  for (const auto& [stride, n] : {std::pair{8, 2}, std::pair{16, 6}}) {
    const int g = kPalmIn / stride;
    for (int y = 0; y < g; ++y)
      for (int x = 0; x < g; ++x)
        for (int k = 0; k < n; ++k) a.emplace_back((x + 0.5) / g, (y + 0.5) / g);
  }
  return a;
}

double area(const Palm& p) { return (p.p1.x - p.p0.x) * (p.p1.y - p.p0.y); }

double iou(const Palm& a, const Palm& b) {
  const double w = std::min(a.p1.x, b.p1.x) - std::max(a.p0.x, b.p0.x);
  const double h = std::min(a.p1.y, b.p1.y) - std::max(a.p0.y, b.p0.y);
  if (w <= 0 || h <= 0) return 0;
  return w * h / (area(a) + area(b) - w * h);
}

// mp_handpose.py _cropAndPadFromPalm: shift, enlarge, crop (clipped) and pad to a square.
struct Crop {
  cv::Mat img;
  cv::Point p0, p1;  // clipped integer box in the source image
  cv::Point bias;    // source coordinate of the padded crop's origin
};

Crop cropAndPad(const cv::Mat& image, cv::Point2d b0, cv::Point2d b1, bool forRotation) {
  cv::Point2d wh = b1 - b0;
  const cv::Point2d shift = forRotation ? cv::Point2d(0, 0) : cv::Point2d(0, -0.4 * wh.y);
  b0 += shift;
  b1 += shift;
  const cv::Point2d c = (b0 + b1) * 0.5;
  wh = b1 - b0;
  const double s = forRotation ? 4.0 : 3.0;
  const cv::Point2d half = wh * (s / 2);
  auto trunc = [](double v) { return static_cast<int>(v); };  // numpy astype(int32)
  Crop r;
  r.p0 = {std::clamp(trunc(c.x - half.x), 0, image.cols), std::clamp(trunc(c.y - half.y), 0, image.rows)};
  r.p1 = {std::clamp(trunc(c.x + half.x), 0, image.cols), std::clamp(trunc(c.y + half.y), 0, image.rows)};
  const int w = std::max(0, r.p1.x - r.p0.x), h = std::max(0, r.p1.y - r.p0.y);
  if (w == 0 || h == 0) return r;
  const cv::Mat roi = image(cv::Rect(r.p0.x, r.p0.y, w, h));
  const int side = forRotation ? static_cast<int>(std::sqrt(double(w) * w + double(h) * h)) : std::max(w, h);
  const int padW = side - w, padH = side - h, left = padW / 2, top = padH / 2;
  cv::copyMakeBorder(roi, r.img, top, padH - top, left, padW - left, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  r.bias = r.p0 - cv::Point(left, top);
  return r;
}

}  // namespace

float HandLandmarks::palmSide() const {
  const cv::Point2f a = pts[5] - pts[0], b = pts[17] - pts[0];
  return a.x * b.y - a.y * b.x;
}

struct HandDetector::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "hand"};
  Ort::SessionOptions so;
  std::unique_ptr<Ort::Session> palm, hand;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<cv::Point2d> anchors = makeAnchors();

  Impl(const std::string& palmOnnx, const std::string& handOnnx, int threads) {
    for (const auto& f : {palmOnnx, handOnnx})
      if (!std::filesystem::exists(f)) throw std::runtime_error("model not found: " + f);
    so.SetIntraOpNumThreads(threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");  // several sessions share the CPU
    palm = std::make_unique<Ort::Session>(env, std::filesystem::path(palmOnnx).c_str(), so);
    hand = std::make_unique<Ort::Session>(env, std::filesystem::path(handOnnx).c_str(), so);
  }

  // mp_palmdet.py: letterbox to 192x192, RGB / 255, decode boxes, NMS
  std::vector<Palm> detectPalms(const cv::Mat& bgr) {
    const int h = bgr.rows, w = bgr.cols;
    const double ratio = std::min(double(kPalmIn) / h, double(kPalmIn) / w);
    cv::Mat img = bgr;
    cv::Point2d padBias(0, 0);
    if (h != kPalmIn || w != kPalmIn) {
      const int rh = static_cast<int>(h * ratio), rw = static_cast<int>(w * ratio);
      cv::resize(bgr, img, {rw, rh});
      const int padW = kPalmIn - rw, padH = kPalmIn - rh, left = padW / 2, top = padH / 2;
      cv::copyMakeBorder(img, img, top, padH - top, left, padW - left, cv::BORDER_CONSTANT, cv::Scalar::all(0));
      padBias = {double(static_cast<int>(left / ratio)), double(static_cast<int>(top / ratio))};
    }
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255);
    const std::array<int64_t, 4> shape{1, kPalmIn, kPalmIn, 3};
    Ort::Value x = Ort::Value::CreateTensor<float>(mem, rgb.ptr<float>(), rgb.total() * 3, shape.data(), 4);
    auto y = palm->Run(Ort::RunOptions{nullptr}, kIn, &x, 1, kPalmOut, 2);
    const float* reg = y[0].GetTensorData<float>();
    const float* cls = y[1].GetTensorData<float>();

    const double scale = std::max(w, h);
    std::vector<Palm> cand;
    for (size_t i = 0; i < anchors.size(); ++i) {
      const double s = 1.0 / (1.0 + std::exp(-double(cls[i])));
      if (s < kScoreThr) continue;
      const float* d = reg + i * 18;
      const cv::Point2d a = anchors[i];
      const cv::Point2d c(d[0] / kPalmIn, d[1] / kPalmIn), wh(d[2] / kPalmIn, d[3] / kPalmIn);
      Palm p;
      p.score = s;
      p.p0 = (c - wh * 0.5 + a) * scale - padBias;
      p.p1 = (c + wh * 0.5 + a) * scale - padBias;
      for (int k = 0; k < 7; ++k)
        p.kp[k] = (cv::Point2d(d[4 + 2 * k] / kPalmIn, d[5 + 2 * k] / kPalmIn) + a) * scale - padBias;
      cand.push_back(p);
    }
    std::sort(cand.begin(), cand.end(), [](const Palm& a, const Palm& b) { return a.score > b.score; });
    std::vector<Palm> keep;
    for (const auto& c : cand)
      if (std::none_of(keep.begin(), keep.end(), [&](const Palm& k) { return iou(c, k) > kNmsThr; })) keep.push_back(c);
    return keep;
  }

  // mp_handpose.py: rotate the palm upright, crop, 224x224, decode landmarks
  HandLandmarks landmarks(const cv::Mat& bgr, const Palm& p) {
    HandLandmarks r;
    r.palmScore = static_cast<float>(p.score);
    r.palmBox = cv::Rect2f(cv::Point2f(p.p0), cv::Point2f(p.p1));

    Crop c1 = cropAndPad(bgr, p.p0, p.p1, true);
    if (c1.img.empty()) return r;
    cv::Mat image;
    cv::cvtColor(c1.img, image, cv::COLOR_BGR2RGB);
    const cv::Point2d padBias(c1.bias);
    const cv::Point2d bb0 = cv::Point2d(c1.p0) - padBias, bb1 = cv::Point2d(c1.p1) - padBias;
    std::array<cv::Point2d, 7> lm;
    for (int k = 0; k < 7; ++k) lm[k] = p.kp[k] - padBias;
    double rad = CV_PI / 2 - std::atan2(-(lm[2].y - lm[0].y), lm[2].x - lm[0].x);
    rad -= 2 * CV_PI * std::floor((rad + CV_PI) / (2 * CV_PI));
    const double angle = rad * 180.0 / CV_PI;
    const cv::Mat M = cv::getRotationMatrix2D((bb0 + bb1) * 0.5, angle, 1.0);  // CV_64F 2x3
    cv::Mat rotated;
    cv::warpAffine(image, rotated, M, image.size());
    cv::Point2d r0(1e18, 1e18), r1(-1e18, -1e18);
    for (const auto& q : lm) {
      const cv::Point2d t(M.at<double>(0, 0) * q.x + M.at<double>(0, 1) * q.y + M.at<double>(0, 2),
                          M.at<double>(1, 0) * q.x + M.at<double>(1, 1) * q.y + M.at<double>(1, 2));
      r0 = {std::min(r0.x, t.x), std::min(r0.y, t.y)};
      r1 = {std::max(r1.x, t.x), std::max(r1.y, t.y)};
    }
    Crop c2 = cropAndPad(rotated, r0, r1, false);
    if (c2.img.empty()) return r;
    cv::Mat blob;
    cv::resize(c2.img, blob, {kHandIn, kHandIn}, 0, 0, cv::INTER_AREA);
    blob.convertTo(blob, CV_32FC3, 1.0 / 255);

    const std::array<int64_t, 4> shape{1, kHandIn, kHandIn, 3};
    Ort::Value x = Ort::Value::CreateTensor<float>(mem, blob.ptr<float>(), blob.total() * 3, shape.data(), 4);
    auto y = hand->Run(Ort::RunOptions{nullptr}, kIn, &x, 1, kHandOut, 4);
    const float* l = y[0].GetTensorData<float>();
    const float* w = y[3].GetTensorData<float>();
    r.presence = y[1].GetTensorData<float>()[0];
    r.handedness = y[2].GetTensorData<float>()[0];
    if (r.presence < kPresenceThr) return r;

    const double s = std::max(double(c2.p1.x - c2.p0.x), double(c2.p1.y - c2.p0.y)) / kHandIn;
    const cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(0, 0), angle, 1.0);
    cv::Mat inv;
    cv::invertAffineTransform(M, inv);
    const cv::Point2d cc = (cv::Point2d(c2.p0) + cv::Point2d(c2.p1)) * 0.5;
    const cv::Point2d oc(inv.at<double>(0, 0) * cc.x + inv.at<double>(0, 1) * cc.y + inv.at<double>(0, 2),
                         inv.at<double>(1, 0) * cc.x + inv.at<double>(1, 1) * cc.y + inv.at<double>(1, 2));
    for (int k = 0; k < 21; ++k) {
      const double lx = (l[3 * k] - kHandIn / 2.0) * s, ly = (l[3 * k + 1] - kHandIn / 2.0) * s;
      // row vector times R[:, :2], as in mp_handpose.py
      const double rx = lx * R.at<double>(0, 0) + ly * R.at<double>(1, 0);
      const double ry = lx * R.at<double>(0, 1) + ly * R.at<double>(1, 1);
      r.pts[k] = cv::Point2f(static_cast<float>(rx + oc.x + padBias.x), static_cast<float>(ry + oc.y + padBias.y));
      // world landmarks get the same in-plane rotation back to the image axes (mp_handpose.py)
      const double wx = w[3 * k], wy = w[3 * k + 1];
      r.world[k] = cv::Point3f(static_cast<float>(wx * R.at<double>(0, 0) + wy * R.at<double>(1, 0)),
                               static_cast<float>(wx * R.at<double>(0, 1) + wy * R.at<double>(1, 1)),
                               w[3 * k + 2]);
    }
    r.found = true;
    return r;
  }
};

HandDetector::HandDetector(const std::string& palmOnnx, const std::string& landmarkOnnx, int threads)
    : p_(std::make_unique<Impl>(palmOnnx, landmarkOnnx, threads)) {}
HandDetector::~HandDetector() = default;

HandLandmarks HandDetector::detect(const cv::Mat& bgr) {
  const auto palms = p_->detectPalms(bgr);
  if (palms.empty()) return {};
  const Palm& best = *std::max_element(palms.begin(), palms.end(), [&](const Palm& a, const Palm& b) { return area(a) < area(b); });
  return p_->landmarks(bgr, best);
}

}  // namespace palm
