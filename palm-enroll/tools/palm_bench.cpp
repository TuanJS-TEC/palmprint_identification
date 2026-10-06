// palm_bench: per-stage timing of the live path, frame -> vector -> match.
//
// Frame:  hand detector, live checks and crop, as palm_gui does on each frame.
// Vector: preprocess (resize 56/224), ONNX inference, L2 normalise.
// Match:  cosine search over the gallery (top-3, as palm_cli identify).
// Input is either full camera frames (--frames DIR) or hand crops (--crops DIR,
// which skips the frame stage). Image decoding is outside the timed region, as
// a camera frame already sits in memory. palm_gui's Qt preview conversion is
// not timed either.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "args.h"
#include "config.h"
#include "gallery.h"
#include "hand_crop.h"
#include "hand_detector.h"
#include "palm_model.h"

namespace fs = std::filesystem;
using namespace palm;

namespace {

const char* kUsage = R"(usage: palm_bench [--config config.yaml] (--frames DIR [--frames-mirrored] | --crops DIR)
                 [--limit N] [--repeat R] [--warmup W] [--gallery-sizes 25,1000,10000] [--gallery FILE]
                 [--det-threads T]
  --frames DIR         full camera frames (*.jpg, *.png): detect -> checks -> crop -> vector -> match
  --crops DIR          hand crops: vector -> match only
  --frames-mirrored    frames are mirror images of the camera view (as palm_eval)
  --repeat R           run over the input R times (default 1)
  --warmup W           drop the first W frames from the statistics (default 5)
  --gallery-sizes L    match against synthetic galleries of these sizes (default 25,1000,10000)
  --gallery FILE       match against a real gallery.bin instead
  --det-threads T      hand detector threads (default 2, as palm_gui)
)";

using Clock = std::chrono::steady_clock;

double msBetween(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Stats {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  double sum() const { return std::accumulate(v.begin(), v.end(), 0.0); }
  double mean() const { return v.empty() ? 0.0 : sum() / static_cast<double>(v.size()); }
  double max() const { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
  double pct(double p) const {  // nearest rank
    if (v.empty()) return 0.0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    const auto k = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(s.size())));
    return s[std::min(s.size(), std::max<size_t>(k, 1)) - 1];
  }
};

void row(const char* name, const Stats& s) {
  if (s.v.empty()) return;
  std::printf("  %-30s %6zu  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f\n", name, s.v.size(), s.mean(), s.pct(50), s.pct(95),
              s.pct(99), s.max());
}

Gallery syntheticGallery(const std::string& version, int dim, int n) {
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd;
  Gallery g(version, dim);
  for (int i = 0; i < n; ++i) {
    Vec v(static_cast<size_t>(dim));
    for (auto& x : v) x = nd(rng);
    g.upsert({"p" + std::to_string(i), 5, l2normalize(v)});
  }
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a(argc, argv, 1, {"frames-mirrored"});
    const bool fromFrames = a.has("frames");
    if (fromFrames == a.has("crops")) {
      std::cerr << kUsage;
      return 1;
    }
    const std::string dir = fromFrames ? a.get("frames") : a.get("crops");
    const Config cfg = loadConfig(a.get("config", "config.yaml"));
    const bool mirrored = a.has("frames-mirrored");
    const int repeat = std::stoi(a.get("repeat", "1"));
    const size_t warmup = std::stoul(a.get("warmup", "5"));
    const size_t limit = std::stoul(a.get("limit", "0"));
    const int detThreads = std::stoi(a.get("det-threads", "2"));

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(dir)) {
      const auto ext = e.path().extension();
      if (e.is_regular_file() && (ext == ".jpg" || ext == ".png")) files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("no .jpg/.png in " + dir);
    if (limit) files.resize(std::min(limit, files.size()));

    PalmModel model(cfg.model);
    const int dim = model.featureDim();

    std::vector<Gallery> galleries;
    std::vector<std::string> galleryNames;
    if (a.has("gallery")) {
      Gallery g(cfg.modelVersion, dim);
      g.load(a.get("gallery"));
      galleryNames.push_back(a.get("gallery") + " (" + std::to_string(g.entries().size()) + " people)");
      galleries.push_back(std::move(g));
    } else {
      std::stringstream ss(a.get("gallery-sizes", "25,1000,10000"));
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        const int n = std::stoi(tok);
        galleries.push_back(syntheticGallery(cfg.modelVersion, dim, n));
        galleryNames.push_back("synthetic " + std::to_string(n));
      }
    }

    std::unique_ptr<HandDetector> det;
    if (fromFrames) det = std::make_unique<HandDetector>(cfg.hand.palmOnnx, cfg.hand.landmarkOnnx, detThreads);

    Stats sDetect, sCheck, sCrop, sPre, sInf, sNorm, sVec;
    std::vector<Stats> sMatch(galleries.size()), sE2e(galleries.size());
    std::vector<double> wallE2e(galleries.size(), 0.0);  // ms, all recorded frames
    double wallFrame = 0;                                // ms, detect + check, all recorded frames
    std::map<HandState, int> rejected;
    size_t idx = 0, nAll = 0, nEmb = 0;

    for (int r = 0; r < repeat; ++r) {
      for (const auto& f : files) {
        const bool record = idx++ >= warmup;
        cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
        if (img.empty()) continue;
        if (fromFrames && mirrored) cv::flip(img, img, 1);

        cv::Mat crop;
        double dDet = 0, dChk = 0, dCrop = 0;
        if (fromFrames) {
          const auto t0 = Clock::now();
          const HandLandmarks h = det->detect(img);
          const auto t1 = Clock::now();
          const HandCheck chk = checkHand(img, h, h, cfg.capture);  // single frame: no motion
          const auto t2 = Clock::now();
          dDet = msBetween(t0, t1);
          dChk = msBetween(t1, t2);
          if (record) {
            ++nAll;
            sDetect.add(dDet);
            sCheck.add(dChk);
            wallFrame += dDet + dChk;
          }
          if (chk.state != HandState::Ok) {
            if (record) {
              ++rejected[chk.state];
              for (auto& w : wallE2e) w += dDet + dChk;
            }
            continue;
          }
          const auto tc = Clock::now();
          crop = img(chk.box);  // palm_gui runs the model on this view, no copy
          dCrop = msBetween(tc, Clock::now());
        } else {
          crop = img;
        }

        // palm_gui runs camera frames unflipped; crops follow mirror_input
        const bool flip = !fromFrames && cfg.mirrorInput;
        const auto tp0 = Clock::now();
        const ModelInputs in = model.preprocess(crop, flip);
        const auto tp1 = Clock::now();
        const Embedding e = model.infer(in);
        const auto tp2 = Clock::now();
        const Vec feat = l2normalize(e.feature);
        const auto tp3 = Clock::now();
        const double dPre = msBetween(tp0, tp1), dInf = msBetween(tp1, tp2), dNorm = msBetween(tp2, tp3);
        const double prefix = dDet + dChk + dCrop + dPre + dInf + dNorm;

        for (size_t k = 0; k < galleries.size(); ++k) {
          const auto tm0 = Clock::now();
          [[maybe_unused]] const auto top = galleries[k].search(feat, 3);
          const double dMatch = msBetween(tm0, Clock::now());
          if (record) {
            sMatch[k].add(dMatch);
            sE2e[k].add(prefix + dMatch);
            wallE2e[k] += prefix + dMatch;
          }
        }
        if (record) {
          if (!fromFrames) ++nAll;  // crops: every file is a sample
          ++nEmb;
          sCrop.add(dCrop);
          sPre.add(dPre);
          sInf.add(dInf);
          sNorm.add(dNorm);
          sVec.add(dPre + dInf + dNorm);
        }
      }
    }

    // ---- report
    const double total = static_cast<double>(nAll);
    std::printf("palm_bench  model %s  threads %d  detector threads %d  input224 %s  mirror_input %d\n",
                cfg.model.onnxPath.c_str(), cfg.model.threads, detThreads,
                cfg.model.input224Original ? "original" : "resize224", cfg.mirrorInput);
    std::printf("source %s (%s): %zu files x %d run(s), warmup %zu, measured %zu\n", dir.c_str(),
                fromFrames ? "full frames" : "hand crops", files.size(), repeat, warmup, nAll);
    if (fromFrames) {
      std::printf("hand accepted by live checks: %zu of %zu (%.1f%%)\n", nEmb, nAll, total > 0 ? 100.0 * nEmb / total : 0.0);
      for (const auto& [s, k] : rejected) std::printf("  rejected: %-42s %d\n", stateMessage(s), k);
    }
    std::printf("\n  %-30s %6s  %8s  %8s  %8s  %8s  %8s   (ms per sample)\n", "stage", "n", "mean", "p50", "p95", "p99",
                "max");
    row("frame: detect (palm+landmarks)", sDetect);
    row("frame: checks (checkHand)", sCheck);
    row("frame: crop", sCrop);
    row("vector: preprocess", sPre);
    row("vector: ONNX inference", sInf);
    row("vector: L2 normalise", sNorm);
    for (size_t k = 0; k < galleries.size(); ++k) row(("match: " + galleryNames[k]).c_str(), sMatch[k]);
    for (size_t k = 0; k < galleries.size(); ++k) row(("frame->match: " + galleryNames[k]).c_str(), sE2e[k]);

    const double vecSum = sVec.sum();
    std::printf("\nthroughput\n");
    std::printf("  vector stage (preprocess+inference+normalise): %.1f vectors/s\n",
                vecSum > 0 ? 1000.0 * static_cast<double>(nEmb) / vecSum : 0.0);
    if (fromFrames)
      std::printf("  frame stage (detect+checks), all frames:        %.1f frames/s\n",
                  wallFrame > 0 ? 1000.0 * total / wallFrame : 0.0);
    for (size_t k = 0; k < galleries.size(); ++k)
      std::printf("  frame->match, all frames, %-22s %.1f frames/s\n", (galleryNames[k] + ":").c_str(),
                  wallE2e[k] > 0 ? 1000.0 * total / wallE2e[k] : 0.0);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
