// palm_handcheck: run the C++ hand detector on labelled camera frames and
// compare with the MediaPipe handedness recorded at capture time.
//   - label agreement (detector vs LABELS.csv)
//   - palm side (cross product sign) per label: tells which label is the
//     physical right hand when all palms face the camera
//   - optional CSV with the 21 landmarks per image (crop calibration)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>

#include <opencv2/imgcodecs.hpp>

#include "args.h"
#include "config.h"
#include "hand_detector.h"
#include "labels.h"

using namespace palm;

namespace {
const char* kUsage = R"(usage: palm_handcheck --labels LABELS.csv --images DIR [--config config.yaml]
                      [--suffix _full.jpg] [--mirrored] [--limit N] [--out hands.csv]
image = DIR/<filename><suffix>. --mirrored: the images are mirror images of the
camera view (output/original is); they are un-mirrored before detection, so the
detector sees what the camera saw.
)";
}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a(argc, argv, 1, {"mirrored"});
    if (!a.has("labels") || !a.has("images")) {
      std::cerr << kUsage;
      return 1;
    }
    const Config cfg = loadConfig(a.get("config", "config.yaml"));
    HandDetector det(cfg.hand.palmOnnx, cfg.hand.landmarkOnnx, cfg.model.threads);
    auto rows = readLabels(a.get("labels"));
    if (a.has("limit")) rows.resize(std::min(rows.size(), std::stoul(a.get("limit"))));
    const std::string suffix = a.get("suffix", "_full.jpg");

    std::ofstream out;
    if (a.has("out")) {
      out.open(a.get("out"));
      out << "filename,csv_label,det_label,handedness,presence,palm_side,width,height";
      for (int k = 0; k < 21; ++k) out << ",x" << k << ",y" << k;
      out << "\n";
    }
    std::map<std::pair<std::string, std::string>, int> agree;  // (csv, detector) -> count
    std::map<std::pair<std::string, int>, int> side;           // (csv, sign) -> count
    int missing = 0, notFound = 0;
    double ms = 0;
    for (const auto& r : rows) {
      cv::Mat img = cv::imread(a.get("images") + "/" + r.file + suffix, cv::IMREAD_COLOR);
      if (img.empty()) {
        ++missing;
        continue;
      }
      if (a.has("mirrored")) cv::flip(img, img, 1);
      const auto t0 = std::chrono::steady_clock::now();
      const HandLandmarks h = det.detect(img);
      ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (!h.found) {
        ++notFound;
        continue;
      }
      ++agree[{r.hand, h.label()}];
      ++side[{r.hand, h.palmSide() < 0 ? -1 : 1}];
      if (out.is_open()) {
        out << r.file << "," << r.hand << "," << h.label() << "," << h.handedness << "," << h.presence << ","
            << h.palmSide() << "," << img.cols << "," << img.rows;
        for (const auto& p : h.pts) out << "," << p.x << "," << p.y;
        out << "\n";
      }
    }
    const int done = static_cast<int>(rows.size()) - missing;
    std::printf("images %d (missing %d), hand found %d, detect %.1f ms/image\n", done, missing, done - notFound,
                ms / std::max(1, done));
    std::printf("label agreement (csv -> detector):\n");
    for (const auto& [k, n] : agree) std::printf("  %-6s -> %-6s %d\n", k.first.c_str(), k.second.c_str(), n);
    std::printf("palm side, cross(index_mcp - wrist, pinky_mcp - wrist) on the camera view:\n");
    for (const auto& [k, n] : side)
      std::printf("  %-6s %s %d\n", k.first.c_str(), k.second < 0 ? "< 0 (right palm / left back)" : "> 0 (left palm / right back)", n);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
