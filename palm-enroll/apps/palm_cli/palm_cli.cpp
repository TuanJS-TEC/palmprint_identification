// palm_cli: embed / enroll / identify / list / remove on hand-crop images.
//
// The images are hand crops (whole hand): a right palm seen from the front,
// thumb on the image's right. Only the right hand is enrolled and identified,
// one entry per person; nothing here checks which hand an image shows.
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
#include "palm/palm.h"

namespace fs = std::filesystem;
using namespace palm;

namespace {

const char* kUsage = R"(usage: palm_cli <command> [--config config.yaml] [options]
  enroll   --id ID [--force] IMAGE...            --force replaces ID or skips the duplicate check
  identify [--top K] IMAGE...                    1:N search
  embed    [--out-dir DIR] IMAGE...              features (.f32) + ROI (.png)
  list
  remove   --id ID
--gallery PATH: use this gallery instead of the one in the config.
)";

struct Sample {
  std::string path;
  Embedding emb;
  Vec feat;  // normalised
  double ms = 0;
};

std::vector<Sample> embedAll(PalmModel& model, const Args& a) {
  std::vector<Sample> out;
  for (const auto& p : a.pos) {
    const cv::Mat bgr = cv::imread(p, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::cerr << "skip (cannot read): " << p << "\n";
      continue;
    }
    const auto t0 = std::chrono::steady_clock::now();
    Sample s{p, model.run(bgr), {}, 0};
    s.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    s.feat = l2normalize(s.emb.feature);
    out.push_back(std::move(s));
  }
  return out;
}

int cmdEmbed(PalmModel& model, const Args& a) {
  const std::string outDir = a.get("out-dir");
  if (!outDir.empty()) fs::create_directories(outDir);
  for (const auto& s : embedAll(model, a)) {
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
  Gallery g(cfg.modelVersion, model.featureDim());
  g.load(cfg.galleryPath);
  const bool exists = g.contains(id);
  if (exists && !a.has("force")) {
    std::printf("enroll %s: REFUSED, already enrolled (one right hand per person); --force to replace\n", id.c_str());
    return 3;
  }
  const auto samples = embedAll(model, a);
  std::vector<Vec> feats;
  for (const auto& s : samples) feats.push_back(s.feat);
  const TemplateResult t = buildTemplate(feats, cfg.tmpl);
  for (size_t i = 0; i < samples.size(); ++i)
    std::printf("  %-60s cos(medoid)=%.3f %s\n", samples[i].path.c_str(), t.cosToMedoid[i],
                std::find(t.kept.begin(), t.kept.end(), static_cast<int>(i)) != t.kept.end() ? "kept" : "OUTLIER");
  if (!t.ok) {
    std::printf("enroll %s: FAILED (%zu/%zu samples kept, need %d)\n", id.c_str(), t.kept.size(), samples.size(),
                cfg.tmpl.minValid);
    return 2;
  }
  const std::vector<Vec> stored = keptSamples(feats, t);
  const auto dup = g.search(stored, 1, id);
  if (!dup.empty() && dup[0].score > cfg.dupCosMax && !a.has("force")) {
    std::printf("enroll %s: REFUSED, matches enrolled '%s' (cos %.3f > dup_cos_max %.3f); --force to override\n",
                id.c_str(), dup[0].id.c_str(), dup[0].score, cfg.dupCosMax);
    return 3;
  }
  g.upsert({id, stored});
  g.save(cfg.galleryPath);
  std::printf("enroll %s: OK%s, right hand, %zu/%zu samples, gallery %s has %zu people\n", id.c_str(),
              exists ? " (replaced)" : "", t.kept.size(), samples.size(), cfg.galleryPath.c_str(), g.entries().size());
  return 0;
}

int cmdIdentify(PalmModel& model, const Config& cfg, const Args& a) {
  Gallery g(cfg.modelVersion, model.featureDim());
  g.load(cfg.galleryPath);
  if (g.entries().empty()) throw std::runtime_error("gallery is empty: " + cfg.galleryPath);
  const int top = std::stoi(a.get("top", "3"));
  for (const auto& s : embedAll(model, a)) {
    const auto m = g.search({s.feat}, top);
    const bool accept = !m.empty() && m[0].score >= cfg.acceptCosMin;
    std::printf("%s  %.1f ms  -> %s", s.path.c_str(), s.ms, accept ? m[0].id.c_str() : "UNKNOWN");
    std::printf("  [");
    for (size_t i = 0; i < m.size(); ++i) std::printf("%s%s %.3f", i ? ", " : "", m[i].id.c_str(), m[i].score);
    std::printf("]  accept_cos_min=%.3f\n", cfg.acceptCosMin);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << kUsage;
    return 1;
  }
  const std::string cmd = argv[1];
  try {
    const Args a(argc, argv, 2, {"force"});
    const std::string configPath = a.get("config", "config.yaml");
    Config cfg = loadConfig(configPath);
    if (a.has("gallery")) cfg.galleryPath = a.get("gallery");
    if (cmd == "list" || cmd == "remove") {
      Gallery g(cfg.modelVersion);
      g.load(cfg.galleryPath);
      if (cmd == "remove") {
        const bool ok = g.remove(a.get("id"));
        if (ok) g.save(cfg.galleryPath);
        std::printf("%s %s\n", a.get("id").c_str(), ok ? "removed" : "not found");
        return ok ? 0 : 2;
      }
      for (const auto& e : g.entries()) std::printf("%-30s %d samples\n", e.id.c_str(), static_cast<int>(e.samples.size()));
      std::printf("%zu people in %s\n", g.entries().size(), cfg.galleryPath.c_str());
      return 0;
    }
    PalmModel model(cfg.model);
    if (cmd == "embed") return cmdEmbed(model, a);
    if (cmd == "enroll") return cmdEnroll(model, cfg, a);
    if (cmd == "identify") return cmdIdentify(model, cfg, a);
    std::cerr << kUsage;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
