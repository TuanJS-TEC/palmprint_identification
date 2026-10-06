// C++ pipeline vs the Python reference from python/export_cpp_ref.py:
//   1. preprocessing (vlResize) vs vl_imread.py
//   2. ONNX Runtime C++ vs ONNX Runtime Python on identical inputs
//   3. the whole C++ path (PNG -> feature)
// usage: test_parity REF_DIR MODEL.onnx   (exit 77 = skipped)
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

#include "check.h"
#include "gallery.h"
#include "palm_model.h"

using namespace palm;
namespace fs = std::filesystem;

static std::vector<float> loadF32(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("missing " + p);
  std::vector<float> v(static_cast<size_t>(f.tellg()) / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
  return v;
}

static cv::Mat loadHwc(const std::string& p, int size) {
  std::vector<float> v = loadF32(p);
  if (v.size() != static_cast<size_t>(size * size * 3)) throw std::runtime_error("bad size " + p);
  return cv::Mat(size, size, CV_32FC3, v.data()).clone();
}

static double maxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = a.size() == b.size() ? 0 : INFINITY;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
  return m;
}

static double maxAbs(const std::vector<float>& a) {
  double m = 0;
  for (float x : a) m = std::max(m, double(std::fabs(x)));
  return m;
}

static std::vector<float> roiChw(const cv::Mat& roi) {
  std::vector<float> v;
  std::vector<cv::Mat> ch;
  cv::split(roi, ch);
  for (const auto& c : ch) v.insert(v.end(), c.begin<float>(), c.end<float>());
  return v;
}

int main(int argc, char** argv) {
  if (argc < 3 || !fs::exists(fs::path(argv[1]) / "manifest.txt") || !fs::exists(argv[2])) {
    std::printf("test_parity: SKIPPED (run python/export_cpp_ref.py; model: %s)\n", argc > 2 ? argv[2] : "?");
    return 77;
  }
  const fs::path ref = argv[1];
  PalmModel model({argv[2], 4, false});
  std::ifstream man(ref / "manifest.txt");
  int cases = 0;
  for (std::string line; std::getline(man, line);) {
    std::istringstream ss(line);
    std::string name;
    int flip = 0;
    if (!(ss >> name >> flip)) continue;
    ++cases;
    const std::string base = (ref / name).string();
    const cv::Mat bgr = cv::imread(base + ".png", cv::IMREAD_COLOR);
    CHECK_MSG(!bgr.empty(), "%s.png", base.c_str());
    if (bgr.empty()) continue;

    // 1. preprocessing
    const ModelInputs in = model.preprocess(bgr, flip != 0);
    const ModelInputs inRef{loadHwc(base + ".x56.f32", 56), loadHwc(base + ".x224.f32", 224)};
    const double d56 = cv::norm(in.x56, inRef.x56, cv::NORM_INF), d224 = cv::norm(in.x224, inRef.x224, cv::NORM_INF);
    CHECK_MSG(d56 < 1e-3 && d224 < 1e-3, "%s: resize max|diff| 56: %g, 224: %g", name.c_str(), d56, d224);

    const auto featRef = loadF32(base + ".feature.f32"), thetaRef = loadF32(base + ".theta.f32");
    const auto roiRef = loadF32(base + ".roi.f32");
    const Vec nRef = l2normalize(featRef);

    // 2. runtime only (identical inputs)
    const Embedding eRt = model.infer(inRef);
    const double rtFeat = maxDiff(eRt.feature, featRef) / maxAbs(featRef);
    CHECK_MSG(rtFeat < 1e-4, "%s: runtime feature rel diff %g", name.c_str(), rtFeat);

    // 3. whole pipeline
    const Embedding e = model.infer(in);
    const std::vector<float> theta(e.theta.begin(), e.theta.end());
    const double dTheta = maxDiff(theta, thetaRef), dRoi = maxDiff(roiChw(e.roi), roiRef);
    const double relFeat = maxDiff(e.feature, featRef) / maxAbs(featRef);
    const float cosv = dot(l2normalize(e.feature), nRef);
    CHECK_MSG(dTheta < 1e-4, "%s: theta max|diff| %g", name.c_str(), dTheta);
    CHECK_MSG(dRoi < 0.05, "%s: roi max|diff| %g (0..255)", name.c_str(), dRoi);
    CHECK_MSG(relFeat < 1e-3 && cosv > 0.99999f, "%s: feature rel diff %g, cosine %.7f", name.c_str(), relFeat, cosv);
    std::printf("  %-6s flip=%d  resize %.1e/%.1e  theta %.1e  roi %.1e  feature rel %.1e (runtime only %.1e)  cos %.7f\n",
                name.c_str(), flip, d56, d224, dTheta, dRoi, relFeat, rtFeat, cosv);
  }
  CHECK(cases > 0);
  return finish("test_parity");
}
