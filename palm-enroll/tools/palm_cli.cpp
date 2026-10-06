// palm_cli: embed / enroll / identify / list / remove on hand-crop images,
// register / check on full camera frames.
//
// embed, enroll and identify take hand crops (whole hand, as in
// output/hand_crop). Only the physical right hand is enrolled and identified,
// one template per person: enroll and identify need the MediaPipe handedness
// label (--label), mapped to the physical hand by hand.right_label in
// config.yaml. register and check go through the C API (palm_api.h) like an
// app would: hand detection, right-palm checks and crop happen inside.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "args.h"
#include "config.h"
#include "gallery.h"
#include "palm_api.h"
#include "palm_model.h"

namespace fs = std::filesystem;
using namespace palm;

namespace {

const char* kUsage = R"(usage: palm_cli <command> [--config config.yaml] [options]
  enroll   --id ID --label Left|Right [--force] IMAGE...  right hand only; --force replaces ID
                                                          or skips the duplicate check
  identify --label Left|Right [--top K] IMAGE...          right hand only, 1:N search
  embed    [--out-dir DIR] IMAGE...                       features (.f32) + ROI (.png)
  list
  remove   --id ID
  register [--mirrored] FRAME...                          full camera frames of one new user -> new id
  check    [--mirrored] FRAME...                          full camera frame -> enrolled id
--label: MediaPipe handedness of the images (the physical right hand is
         hand.right_label in config.yaml). --flip: mirror the images first.
--mirrored: the frames are mirror images of the camera view (output/original).
--gallery PATH: use this gallery instead of the one in the config.
)";

// enroll / identify accept the physical right hand only
void requireRightHand(const Config& cfg, const Args& a) {
  if (!a.has("label")) throw std::runtime_error("--label Left|Right (MediaPipe handedness) is required");
  const std::string label = a.get("label");
  if (label != "Left" && label != "Right") throw std::runtime_error("--label must be Left or Right");
  if (!cfg.isRightHand(label))
    throw std::runtime_error("only the right hand can be used: MediaPipe label '" + label +
                             "' is the left hand in this setup (hand.right_label: " + cfg.rightLabel + ")");
}

struct Sample {
  std::string path;
  Embedding emb;
  Vec feat;  // normalised
  double ms = 0;
};

std::vector<Sample> embedAll(PalmModel& model, const Config& cfg, const Args& a) {
  const bool flip = a.has("flip") || cfg.mirrorInput;
  std::vector<Sample> out;
  for (const auto& p : a.pos) {
    const cv::Mat bgr = cv::imread(p, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::cerr << "skip (cannot read): " << p << "\n";
      continue;
    }
    const auto t0 = std::chrono::steady_clock::now();
    Sample s{p, model.run(bgr, flip), {}, 0};
    s.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    s.feat = l2normalize(s.emb.feature);
    out.push_back(std::move(s));
  }
  return out;
}

int cmdEmbed(PalmModel& model, const Config& cfg, const Args& a) {
  const std::string outDir = a.get("out-dir");
  if (!outDir.empty()) fs::create_directories(outDir);
  for (const auto& s : embedAll(model, cfg, a)) {
    std::printf("%s  %.1f ms  |f|=%.3f  theta x=[", s.path.c_str(), s.ms,
                std::sqrt(dot(s.emb.feature, s.emb.feature)));
    for (int i = 0; i < 9; ++i) std::printf("%s%.2f", i ? " " : "", s.emb.theta[i]);
    std::printf("] y=[");
    for (int i = 9; i < 18; ++i) std::printf("%s%.2f", i > 9 ? " " : "", s.emb.theta[i]);
    std::printf("]\n");
    if (outDir.empty()) continue;
    const std::string stem = (fs::path(outDir) / fs::path(s.path).stem()).string();
    std::ofstream(stem + ".f32", std::ios::binary)
        .write(reinterpret_cast<const char*>(s.emb.feature.data()), s.emb.feature.size() * sizeof(float));
    cv::Mat roi8;
    s.emb.roi.convertTo(roi8, CV_8UC3);
    cv::cvtColor(roi8, roi8, cv::COLOR_RGB2BGR);
    cv::imwrite(stem + "_roi.png", roi8);
  }
  return 0;
}

int cmdEnroll(PalmModel& model, const Config& cfg, const Args& a) {
  const std::string id = a.get("id");
  if (id.empty()) throw std::runtime_error("enroll: --id is required");
  requireRightHand(cfg, a);
  Gallery g(cfg.modelVersion, model.featureDim());
  g.load(cfg.galleryPath);
  const bool exists = std::any_of(g.entries().begin(), g.entries().end(), [&](const GalleryEntry& e) { return e.id == id; });
  if (exists && !a.has("force")) {
    std::printf("enroll %s: REFUSED, already enrolled (one right hand per person); --force to replace\n", id.c_str());
    return 3;
  }
  const auto samples = embedAll(model, cfg, a);
  std::vector<Vec> feats;
  for (const auto& s : samples) feats.push_back(s.feat);
  const TemplateResult t = buildTemplate(feats, cfg.tmpl);
  for (size_t i = 0; i < samples.size(); ++i)
    std::printf("  %-60s cos(medoid)=%.3f %s\n", samples[i].path.c_str(), t.cosToMedoid[i],
                std::find(t.kept.begin(), t.kept.end(), static_cast<int>(i)) != t.kept.end() ? "kept" : "OUTLIER");
  if (!t.ok) {
    std::printf("enroll %s: FAILED (%s, %zu kept, need %d)\n", id.c_str(), t.reason.c_str(), t.kept.size(),
                cfg.tmpl.minValid);
    return 2;
  }
  const auto dup = g.search(t.tmpl, 1, id);
  if (!dup.empty() && dup[0].score > cfg.dupCosMax && !a.has("force")) {
    std::printf("enroll %s: REFUSED, matches enrolled '%s' (cos %.3f > dup_cos_max %.3f); --force to override\n",
                id.c_str(), dup[0].id.c_str(), dup[0].score, cfg.dupCosMax);
    return 3;
  }
  g.upsert({id, static_cast<int>(t.kept.size()), t.tmpl});
  g.save(cfg.galleryPath);
  std::printf("enroll %s: OK%s, right hand, %zu/%zu samples, gallery %s has %zu people\n", id.c_str(),
              exists ? " (replaced)" : "", t.kept.size(), samples.size(), cfg.galleryPath.c_str(), g.entries().size());
  return 0;
}

int cmdIdentify(PalmModel& model, const Config& cfg, const Args& a) {
  requireRightHand(cfg, a);
  Gallery g(cfg.modelVersion, model.featureDim());
  g.load(cfg.galleryPath);
  if (g.entries().empty()) throw std::runtime_error("gallery is empty: " + cfg.galleryPath);
  const int top = std::stoi(a.get("top", "3"));
  for (const auto& s : embedAll(model, cfg, a)) {
    const auto m = g.search(s.feat, top);
    const bool accept = !m.empty() && m[0].score >= cfg.acceptCosMin;
    std::printf("%s  %.1f ms  -> %s", s.path.c_str(), s.ms, accept ? m[0].id.c_str() : "UNKNOWN");
    std::printf("  [");
    for (size_t i = 0; i < m.size(); ++i) std::printf("%s%s %.3f", i ? ", " : "", m[i].id.c_str(), m[i].score);
    std::printf("]  accept_cos_min=%.3f\n", cfg.acceptCosMin);
  }
  return 0;
}

// ---- register / check through the C API
struct Api {
  PalmHandle h = nullptr;
  Api(const std::string& configPath, const std::string& galleryPath) {
    PalmInitOptions o{};
    o.config_path = configPath.c_str();
    o.gallery_path = galleryPath.c_str();
    if (palm_init(&o, &h) != PALM_OK) throw std::runtime_error(palm_last_error(nullptr));
  }
  ~Api() { palm_release(h); }
  Api(const Api&) = delete;
  Api& operator=(const Api&) = delete;
};

PalmFrame bgrFrame(const cv::Mat& m, bool mirrored) {
  PalmFrame f{};
  f.data = m.data;
  f.size = m.step * m.rows;
  f.width = m.cols;
  f.height = m.rows;
  f.stride = static_cast<int>(m.step);
  f.format = PALM_FMT_BGR;
  f.mirrored = mirrored;
  return f;
}

int cmdRegister(Api& api, const Args& a) {
  std::vector<std::string> names;
  std::vector<cv::Mat> imgs;
  for (const auto& p : a.pos) {
    cv::Mat m = cv::imread(p, cv::IMREAD_COLOR);
    if (m.empty()) {
      std::cerr << "skip (cannot read): " << p << "\n";
      continue;
    }
    names.push_back(p);
    imgs.push_back(std::move(m));
  }
  if (imgs.empty()) throw std::runtime_error("register: no readable frame");
  std::vector<PalmFrame> frames;
  for (const auto& m : imgs) frames.push_back(bgrFrame(m, a.has("mirrored")));
  std::vector<int> status(frames.size(), -1);
  char id[PALM_ID_MAX];
  float score = 0;
  const int s =
      palm_register(api.h, frames.data(), static_cast<int>(frames.size()), id, sizeof id, &score, status.data());
  for (size_t i = 0; i < names.size(); ++i)
    if (status[i] >= 0) std::printf("  %-60s %s\n", names[i].c_str(), palm_status_text(status[i]));
  if (s == PALM_OK) {
    std::printf("register: OK %s (closest other user: cos %.3f)\n", id, score);
    return 0;
  }
  if (s == PALM_ERR_ALREADY_REGISTERED) {
    std::printf("register: ALREADY_REGISTERED as %s (cos %.3f)\n", id, score);
    return 3;
  }
  std::printf("register: FAILED, %s: %s\n", palm_status_text(s), palm_last_error(api.h));
  return 2;
}

int cmdCheck(Api& api, const Args& a) {
  int failed = 0;
  for (const auto& p : a.pos) {
    const cv::Mat m = cv::imread(p, cv::IMREAD_COLOR);
    if (m.empty()) {
      std::cerr << "skip (cannot read): " << p << "\n";
      ++failed;
      continue;
    }
    const PalmFrame f = bgrFrame(m, a.has("mirrored"));
    char id[PALM_ID_MAX];
    float score = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const int s = palm_check_register(api.h, &f, id, sizeof id, &score);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (s == PALM_OK) {
      std::printf("%s  %.1f ms  -> %s  cos %.3f\n", p.c_str(), ms, id, score);
    } else {
      std::printf("%s  %.1f ms  -> %s (%s)\n", p.c_str(), ms, palm_status_text(s), palm_last_error(api.h));
      ++failed;
    }
  }
  return failed ? 2 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << kUsage;
    return 1;
  }
  const std::string cmd = argv[1];
  try {
    const Args a(argc, argv, 2, {"flip", "force", "mirrored"});
    const std::string configPath = a.get("config", "config.yaml");
    Config cfg = loadConfig(configPath);
    if (a.has("gallery")) cfg.galleryPath = a.get("gallery");
    if (cmd == "register" || cmd == "check") {
      Api api(configPath, cfg.galleryPath);
      return cmd == "register" ? cmdRegister(api, a) : cmdCheck(api, a);
    }
    if (cmd == "list" || cmd == "remove") {
      Gallery g(cfg.modelVersion);
      g.load(cfg.galleryPath);
      if (cmd == "remove") {
        const bool ok = g.remove(a.get("id"));
        if (ok) g.save(cfg.galleryPath);
        std::printf("%s %s\n", a.get("id").c_str(), ok ? "removed" : "not found");
        return ok ? 0 : 2;
      }
      for (const auto& e : g.entries()) std::printf("%-30s %d samples\n", e.id.c_str(), e.numSamples);
      std::printf("%zu people in %s\n", g.entries().size(), cfg.galleryPath.c_str());
      return 0;
    }
    PalmModel model(cfg.model);
    if (cmd == "embed") return cmdEmbed(model, cfg, a);
    if (cmd == "enroll") return cmdEnroll(model, cfg, a);
    if (cmd == "identify") return cmdIdentify(model, cfg, a);
    std::cerr << kUsage;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
