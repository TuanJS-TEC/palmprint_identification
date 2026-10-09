// palm_profile: the per-frame work split finer than palm_bench does.
//
// Detector: HandDetector::detect as a whole, plus Ort::Session::Run alone for
//   the palm detector and the landmark model, on sessions created with the
//   library's options (pre/post-processing = detect minus the two runs). The
//   palm input is letterboxed as in hand_detector.cpp; the landmark input is
//   any 224x224 image, as the network's cost does not depend on the content.
// EE-PRnet: PalmModel::preprocess + infer redone step by step with the library
//   functions (BGR->RGB float, the two vl_imreadjpeg resizes, HWC->CHW,
//   Session::Run, output copy, L2 normalisation, gallery search). The feature
//   is compared with PalmModel::run on the first frames.
// --ort-profile PREFIX: ONNX Runtime per-node profile of EE-PRnet over the
//   accepted frames, written as PREFIX_<date>.json for
//   python/ort_profile_groups.py.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "args.h"
#include "config.h"
#include "gallery.h"
#include "hand_crop.h"
#include "hand_detector.h"
#include "palm_model.h"
#include "preprocess.h"

namespace fs = std::filesystem;
using namespace palm;

namespace {

const char* kUsage = R"(usage: palm_profile [--config config.yaml] --frames DIR [--frames-mirrored] [--limit N]
                    [--warmup W] [--det-threads 2] [--gallery-size 1000]
                    [--ort-profile PREFIX [--profile-frames 50] [--profile-repeat 2]]
)";

const char* kEeIn[] = {"input_56", "input_224"};
const char* kEeOut[] = {"feature", "theta", "roi"};
const char* kMpIn[] = {"input_1"};
const char* kPalmOut[] = {"Identity", "Identity_1"};
const char* kHandOut[] = {"Identity", "Identity_1", "Identity_2"};
constexpr int kPalmIn = 192, kHandIn = 224;

using Clock = std::chrono::steady_clock;

double msBetween(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Stats {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  double mean() const { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }
  double pct(double p) const {  // nearest rank
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    const auto k = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(s.size())));
    return s[std::min(s.size(), std::max<size_t>(k, 1)) - 1];
  }
  double max() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
};

void row(const char* name, const Stats& s) {
  std::printf("  %-44s %6zu  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n", name, s.v.size(), s.mean(), s.pct(50), s.pct(95),
              s.pct(99), s.max());
}

// same options as PalmModel and HandDetector
Ort::Session makeSession(Ort::Env& env, const std::string& path, int threads, const std::string& profilePrefix = {}) {
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(threads);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  so.AddConfigEntry("session.intra_op.allow_spinning", "0");
  if (!profilePrefix.empty()) so.EnableProfiling(profilePrefix.c_str());
  return Ort::Session(env, fs::path(path).c_str(), so);
}

// hand_detector.cpp detectPalms: letterbox to 192x192, RGB float / 255
cv::Mat palmInput(const cv::Mat& bgr) {
  const int h = bgr.rows, w = bgr.cols;
  const double ratio = std::min(double(kPalmIn) / h, double(kPalmIn) / w);
  cv::Mat img;
  const int rh = static_cast<int>(h * ratio), rw = static_cast<int>(w * ratio);
  cv::resize(bgr, img, {rw, rh});
  const int padW = kPalmIn - rw, padH = kPalmIn - rh, left = padW / 2, top = padH / 2;
  cv::copyMakeBorder(img, img, top, padH - top, left, padW - left, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  cv::Mat rgb;
  cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255);
  return rgb;
}

Gallery syntheticGallery(const std::string& version, int dim, int n) {
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd;
  Gallery g(version, dim);
  for (int i = 0; i < n; ++i) {
    Vec v(static_cast<size_t>(dim));
    for (auto& x : v) x = nd(rng);
    g.upsert({"p" + std::to_string(i), {l2normalize(v)}});
  }
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a(argc, argv, 1, {"frames-mirrored"});
    if (!a.has("frames")) {
      std::cerr << kUsage;
      return 1;
    }
    const Config cfg = loadConfig(a.get("config", "config.yaml"));
    if (cfg.model.input224Original) throw std::runtime_error("palm_profile assumes model.input224_mode resize224");
    const size_t limit = std::stoul(a.get("limit", "0")), warmup = std::stoul(a.get("warmup", "5"));
    const int detThreads = std::stoi(a.get("det-threads", "2"));
    const std::string profilePrefix = a.get("ort-profile");
    const size_t profileFrames = std::stoul(a.get("profile-frames", "50"));
    const int profileRepeat = std::stoi(a.get("profile-repeat", "2"));

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(a.get("frames"))) {
      const auto ext = e.path().extension();
      if (e.is_regular_file() && (ext == ".jpg" || ext == ".png")) files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (limit) files.resize(std::min(limit, files.size()));

    PalmModel model(cfg.model);  // the library path, for the parity check
    HandDetector det(cfg.hand.palmOnnx, cfg.hand.landmarkOnnx, detThreads);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "profile");
    Ort::Session ee = makeSession(env, cfg.model.onnxPath, cfg.model.threads);
    Ort::Session palmS = makeSession(env, cfg.hand.palmOnnx, detThreads);
    Ort::Session handS = makeSession(env, cfg.hand.landmarkOnnx, detThreads);
    const Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const Gallery gallery = syntheticGallery(cfg.modelVersion, model.featureDim(), std::stoi(a.get("gallery-size", "1000")));

    Stats sDet, sPalmPre, sPalmRun, sHandRun, sCheck, sCrop, sRgb, sR56, sR224, sPack, sRun, sOut, sNorm, sMatch;
    Stats sModels, sE2e;
    std::vector<std::pair<std::vector<float>, std::vector<float>>> stored;  // EE-PRnet inputs for --ort-profile
    float maxDiff = 0;
    int nParity = 0;
    size_t idx = 0, nFrames = 0, nEmb = 0;

    for (const auto& f : files) {
      cv::Mat frame = cv::imread(f, cv::IMREAD_COLOR);
      if (frame.empty()) continue;
      if (a.has("frames-mirrored")) cv::flip(frame, frame, 1);
      const bool record = idx++ >= warmup;

      // ---- frame stage
      const auto t0 = Clock::now();
      const HandLandmarks h = det.detect(frame);
      const auto t1 = Clock::now();
      const HandCheck chk = checkHand(frame, h, h, cfg.capture);  // single frame: no motion
      const auto t2 = Clock::now();

      // pure model time of the two detector networks
      const auto p0 = Clock::now();
      cv::Mat pin = palmInput(frame);
      const auto p1 = Clock::now();
      const std::array<int64_t, 4> palmShape{1, kPalmIn, kPalmIn, 3}, handShape{1, kHandIn, kHandIn, 3};
      Ort::Value px = Ort::Value::CreateTensor<float>(mem, pin.ptr<float>(), pin.total() * 3, palmShape.data(), 4);
      const auto p2 = Clock::now();
      palmS.Run(Ort::RunOptions{nullptr}, kMpIn, &px, 1, kPalmOut, 2);
      const auto p3 = Clock::now();
      double dHand = 0;
      if (h.palmScore > 0) {  // detect ran the landmark model: a palm was found
        cv::Mat hin;
        cv::resize(frame, hin, {kHandIn, kHandIn}, 0, 0, cv::INTER_AREA);
        hin.convertTo(hin, CV_32FC3, 1.0 / 255);
        Ort::Value hx = Ort::Value::CreateTensor<float>(mem, hin.ptr<float>(), hin.total() * 3, handShape.data(), 4);
        const auto h0 = Clock::now();
        handS.Run(Ort::RunOptions{nullptr}, kMpIn, &hx, 1, kHandOut, 3);
        dHand = msBetween(h0, Clock::now());
        if (record) sHandRun.add(dHand);
      }
      if (record) {
        ++nFrames;
        sDet.add(msBetween(t0, t1));
        sCheck.add(msBetween(t1, t2));
        sPalmPre.add(msBetween(p0, p1));
        sPalmRun.add(msBetween(p2, p3));
      }
      if (chk.state != HandState::Ok) continue;

      // ---- vector stage, step by step
      const auto q0 = Clock::now();
      const cv::Mat crop = frame(chk.box);  // a view, as in palm_gui
      const auto q1 = Clock::now();
      const cv::Mat rgb = toRgbFloat(crop, false);
      const auto q2 = Clock::now();
      const cv::Mat x56 = vlResize(rgb, {56, 56});
      const auto q3 = Clock::now();
      const cv::Mat x224 = vlResize(rgb, {224, 224});
      const auto q4 = Clock::now();
      std::vector<float> va(3 * x56.total()), vb(3 * x224.total());
      packChw(x56, va.data());
      packChw(x224, vb.data());
      const auto q5 = Clock::now();
      const std::array<int64_t, 4> sa{1, 3, 56, 56}, sb{1, 3, 224, 224};
      std::array<Ort::Value, 2> x{Ort::Value::CreateTensor<float>(mem, va.data(), va.size(), sa.data(), 4),
                                  Ort::Value::CreateTensor<float>(mem, vb.data(), vb.size(), sb.data(), 4)};
      const auto q6 = Clock::now();
      auto y = ee.Run(Ort::RunOptions{nullptr}, kEeIn, x.data(), 2, kEeOut, 3);
      const auto q7 = Clock::now();
      // output copy as PalmModel::infer
      Embedding e;
      const float* fp = y[0].GetTensorData<float>();
      e.feature.assign(fp, fp + y[0].GetTensorTypeAndShapeInfo().GetElementCount());
      const float* tp = y[1].GetTensorData<float>();
      std::copy(tp, tp + 18, e.theta.begin());
      const float* rp = y[2].GetTensorData<float>();
      std::vector<cv::Mat> planes;
      for (int c = 0; c < 3; ++c) planes.emplace_back(112, 112, CV_32F, const_cast<float*>(rp + c * 112 * 112));
      cv::merge(planes, e.roi);
      const auto q8 = Clock::now();
      const Vec feat = l2normalize(e.feature);
      const auto q9 = Clock::now();
      [[maybe_unused]] const auto top = gallery.search(feat, 2);
      const auto q10 = Clock::now();

      if (nParity < 3) {  // the step-by-step path must give the library's feature
        const Embedding ref = model.run(crop, false);
        for (size_t i = 0; i < ref.feature.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(ref.feature[i] - e.feature[i]));
        ++nParity;
      }
      if (!profilePrefix.empty() && stored.size() < profileFrames) stored.emplace_back(va, vb);
      if (!record) continue;
      ++nEmb;
      sCrop.add(msBetween(q0, q1));
      sRgb.add(msBetween(q1, q2));
      sR56.add(msBetween(q2, q3));
      sR224.add(msBetween(q3, q4));
      sPack.add(msBetween(q4, q5));
      sRun.add(msBetween(q6, q7));
      sOut.add(msBetween(q7, q8));
      sNorm.add(msBetween(q8, q9));
      sMatch.add(msBetween(q9, q10));
      sModels.add(msBetween(p2, p3) + dHand + msBetween(q6, q7));
      sE2e.add(msBetween(t0, t2) + msBetween(q0, q10));
    }

    std::printf("palm_profile  model threads %d  detector threads %d  gallery %zu\n", cfg.model.threads, detThreads,
                gallery.entries().size());
    std::printf("frames %zu measured (warm-up %zu), %zu passed the live checks\n", nFrames, warmup, nEmb);
    std::printf("step-by-step EE-PRnet vs PalmModel::run: max |feature diff| %.3g over %d frames\n", maxDiff, nParity);
    std::printf("\n  %-44s %6s  %8s  %8s  %8s  %8s  %8s   (ms)\n", "step", "n", "mean", "p50", "p95", "p99", "max");
    row("detect: HandDetector::detect (total)", sDet);
    row("detect: palm input letterbox+RGB+float", sPalmPre);
    row("detect: palm detector Session::Run", sPalmRun);
    row("detect: landmark model Session::Run", sHandRun);
    row("checks: checkHand (box, side, sharpness)", sCheck);
    row("crop: hand box view", sCrop);
    row("vector: BGR->RGB + uint8->float", sRgb);
    row("vector: vl resize 56x56", sR56);
    row("vector: vl resize 224x224", sR224);
    row("vector: HWC->CHW packing (both inputs)", sPack);
    row("vector: EE-PRnet Session::Run", sRun);
    row("vector: output copy (feature, theta, roi)", sOut);
    row("vector: L2 normalise", sNorm);
    row("match: gallery search", sMatch);
    row("all three networks' Session::Run", sModels);
    row("frame -> match (sum of the steps above)", sE2e);

    if (!profilePrefix.empty()) {
      Ort::Session prof = makeSession(env, cfg.model.onnxPath, cfg.model.threads, profilePrefix);
      const std::array<int64_t, 4> sa{1, 3, 56, 56}, sb{1, 3, 224, 224};
      for (int r = 0; r < profileRepeat; ++r)
        for (auto& [va, vb] : stored) {
          std::array<Ort::Value, 2> x{Ort::Value::CreateTensor<float>(mem, va.data(), va.size(), sa.data(), 4),
                                      Ort::Value::CreateTensor<float>(mem, vb.data(), vb.size(), sb.data(), 4)};
          prof.Run(Ort::RunOptions{nullptr}, kEeIn, x.data(), 2, kEeOut, 3);
        }
      Ort::AllocatorWithDefaultOptions alloc;
      std::printf("\nORT profile (%zu runs): %s\n", stored.size() * profileRepeat,
                  prof.EndProfilingAllocated(alloc).get());
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
