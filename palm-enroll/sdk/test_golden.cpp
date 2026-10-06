// Golden test of the palm SDK through its C API (palm_api.h): runs a fixed
// scenario on testdata/ and compares with the results recorded on the
// reference machine (x86-64 Linux, ONNX Runtime 1.16.3, OpenCV 4.10).
// Run it on the phone too, to check that a port gives the same results.
//
//   test_golden MODELS_DIR TESTDATA_DIR
//       verify; exit 0 when every step matches
//   test_golden MODELS_DIR TESTDATA_DIR --generate A1 A2 A3 A4 A5 A_PROBE STRANGER LEFT
//       (maintainers) source frames, mirror images of the camera view: writes
//       them un-mirrored as testdata/*.png, then records expected.txt and features/
//
// Frames go in as RGBA buffers with mirrored = 0, as a phone camera gives them.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "palm_api.h"

namespace fs = std::filesystem;

namespace {

const char* kFrames[] = {"a_1", "a_2", "a_3", "a_4", "a_5", "a_probe", "stranger", "left"};

struct Result {
  int status = 0;
  std::string id = "-";
  float score = 0.f;
};

std::string modelPath(const std::string& dir, const char* name) { return (fs::path(dir) / name).string(); }

PalmFrame frameOf(const cv::Mat& m, int format, int rotation = 0) {
  PalmFrame f{};
  f.data = m.data;
  f.size = m.step * m.rows;
  f.width = m.cols;
  f.height = m.rows;
  f.stride = static_cast<int>(m.step);
  f.format = format;
  f.rotation = rotation;
  return f;
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i], na += a[i] * a[i], nb += b[i] * b[i];
  return static_cast<float>(s / std::sqrt(na * nb + 1e-30));
}

std::vector<unsigned char> toNv21(const cv::Mat& bgr) {
  cv::Mat i420;
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
  const int w = bgr.cols, h = bgr.rows, q = (w / 2) * (h / 2);
  std::vector<unsigned char> out(i420.data, i420.data + w * h);
  const unsigned char* u = i420.data + w * h;
  const unsigned char* v = u + q;
  for (int k = 0; k < q; ++k) out.push_back(v[k]), out.push_back(u[k]);
  return out;
}

int fails = 0;
void expect(bool ok, const std::string& what) {
  std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  fails += !ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: test_golden MODELS_DIR TESTDATA_DIR [--generate A1 A2 A3 A4 A5 A_PROBE STRANGER LEFT]\n");
    return 2;
  }
  const std::string models = argv[1], data = argv[2];
  const bool generate = argc > 3 && std::strcmp(argv[3], "--generate") == 0;
  const int nFrames = static_cast<int>(sizeof kFrames / sizeof *kFrames);
  if (generate) {
    if (argc != 4 + nFrames) {
      std::fprintf(stderr, "--generate needs %d source frames\n", nFrames);
      return 2;
    }
    fs::create_directories(fs::path(data) / "features");
    for (int i = 0; i < nFrames; ++i) {
      cv::Mat m = cv::imread(argv[4 + i], cv::IMREAD_COLOR);
      if (m.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[4 + i]);
        return 2;
      }
      cv::flip(m, m, 1);  // the capture log stores mirror images; a camera buffer is not mirrored
      cv::imwrite((fs::path(data) / (std::string(kFrames[i]) + ".png")).string(), m);
    }
  }

  // ---- init as an app without a config file: explicit model paths, built-in thresholds
  const std::string galleryPath = (fs::path(data) / "golden_gallery.bin").string();
  fs::remove(galleryPath);
  const std::string embed = modelPath(models, "EE_PRnet-epoch-60.onnx");
  const std::string palmDet = modelPath(models, "palm_detection_mediapipe_2023feb.onnx");
  const std::string landmark = modelPath(models, "handpose_estimation_mediapipe_2023feb.onnx");
  PalmInitOptions opt{};
  opt.embed_onnx = embed.c_str();
  opt.palm_onnx = palmDet.c_str();
  opt.landmark_onnx = landmark.c_str();
  opt.gallery_path = galleryPath.c_str();
  PalmHandle h = nullptr;
  if (palm_init(&opt, &h) != PALM_OK) {
    std::fprintf(stderr, "palm_init: %s\n", palm_last_error(nullptr));
    return 1;
  }

  // ---- load the frames as RGBA, like a CameraX RGBA_8888 buffer
  std::map<std::string, cv::Mat> bgr, rgba;
  for (const char* name : kFrames) {
    bgr[name] = cv::imread((fs::path(data) / (std::string(name) + ".png")).string(), cv::IMREAD_COLOR);
    if (bgr[name].empty()) {
      std::fprintf(stderr, "missing %s/%s.png\n", data.c_str(), name);
      return 1;
    }
    cv::cvtColor(bgr[name], rgba[name], cv::COLOR_BGR2RGBA);
  }

  std::map<std::string, Result> got;
  std::map<std::string, std::vector<float>> feat;
  char id[PALM_ID_MAX];
  float score = 0;

  // palm_extract on every frame (timed: hand detection + embedding)
  double ms = 0;
  int timed = 0;
  for (const char* name : kFrames) {
    std::vector<float> f(PALM_FEATURE_DIM, 0.f);
    PalmFrame fr = frameOf(rgba[name], PALM_FMT_RGBA);
    const auto t0 = std::chrono::steady_clock::now();
    const int s = palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM);
    if (s == PALM_OK) ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), ++timed;
    got[std::string("extract.") + name] = {s, "-", 0.f};
    feat[name] = f;
  }
  std::printf("palm_extract: %.0f ms per frame with a palm (mean of %d; the first call includes warm-up)\n",
              timed ? ms / timed : 0.0, timed);
  // palm_register: five frames of A, then A again, which must be refused
  const auto registerFrames = [&](const std::vector<std::string>& names) {
    std::vector<PalmFrame> frames;
    for (const auto& n : names) frames.push_back(frameOf(rgba[n], PALM_FMT_RGBA));
    const int s = palm_register(h, frames.data(), static_cast<int>(frames.size()), id, sizeof id, &score, nullptr);
    return Result{s, id[0] ? id : "-", score};
  };
  got["register.a"] = registerFrames({"a_1", "a_2", "a_3", "a_4", "a_5"});
  got["register.a_again"] = registerFrames({"a_2", "a_3", "a_4", "a_5", "a_probe"});
  // palm_check_register: A's other frame, a stranger, a left hand, an empty frame
  for (const char* name : {"a_probe", "stranger", "left"}) {
    PalmFrame fr = frameOf(rgba[name], PALM_FMT_RGBA);
    const int s = palm_check_register(h, &fr, id, sizeof id, &score);
    got[std::string("check.") + name] = {s, id[0] ? id : "-", score};
  }
  const cv::Mat black(720, 1280, CV_8UC4, cv::Scalar::all(0));
  PalmFrame blackFrame = frameOf(black, PALM_FMT_RGBA);
  got["check.black"] = {palm_check_register(h, &blackFrame, id, sizeof id, &score), "-", 0.f};
  // palm_compare of two frames of A
  int same = 0;
  const int cs = palm_compare(h, feat["a_1"].data(), feat["a_probe"].data(), PALM_FEATURE_DIM, &score, &same);
  got["compare.a_1.a_probe"] = {cs, same ? "same" : "different", score};

  // the same pixels in other buffer layouts give the same feature (checked here, not recorded)
  std::printf("buffer formats (vs RGBA):\n");
  std::vector<float> f(PALM_FEATURE_DIM);
  PalmFrame fr = frameOf(bgr["a_1"], PALM_FMT_BGR);
  expect(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM) == PALM_OK && cosine(f, feat["a_1"]) > 0.9999f, "BGR");
  cv::Mat sensor;  // a buffer that needs a 90 degree clockwise turn, as on most phones
  cv::rotate(rgba["a_1"], sensor, cv::ROTATE_90_COUNTERCLOCKWISE);
  fr = frameOf(sensor, PALM_FMT_RGBA, 90);
  expect(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM) == PALM_OK && cosine(f, feat["a_1"]) > 0.9999f,
         "RGBA rotation 90");
  const std::vector<unsigned char> nv21 = toNv21(bgr["a_1"]);
  PalmFrame yuv{nv21.data(), nv21.size(), bgr["a_1"].cols, bgr["a_1"].rows, 0, PALM_FMT_NV21, 0, 0};
  const int ns = palm_extract(h, &yuv, f.data(), PALM_FEATURE_DIM);
  expect(ns == PALM_OK && cosine(f, feat["a_1"]) > 0.98f, "NV21 (lossy chroma, cos " + std::to_string(cosine(f, feat["a_1"])) + ")");
  palm_release(h);
  fs::remove(galleryPath);

  const fs::path expectedPath = fs::path(data) / "expected.txt";
  if (generate) {
    std::ofstream out(expectedPath);
    out << "# step status id score   (recorded by test_golden --generate)\n";
    for (const auto& [k, r] : got) out << k << ' ' << r.status << ' ' << r.id << ' ' << r.score << '\n';
    for (const char* name : kFrames)
      if (got[std::string("extract.") + name].status == PALM_OK)
        std::ofstream(fs::path(data) / "features" / (std::string(name) + ".f32"), std::ios::binary)
            .write(reinterpret_cast<const char*>(feat[name].data()), PALM_FEATURE_DIM * sizeof(float));
    std::printf("recorded %zu steps in %s\n", got.size(), expectedPath.string().c_str());
    return fails ? 1 : 0;
  }

  // ---- compare with the recorded results
  std::ifstream in(expectedPath);
  if (!in) {
    std::fprintf(stderr, "missing %s\n", expectedPath.string().c_str());
    return 1;
  }
  std::printf("scenario (status and id must match, score within 0.01, features cos >= 0.999):\n");
  std::string line;
  int steps = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string key;
    Result e;
    ss >> key >> e.status >> e.id >> e.score;
    ++steps;
    const auto it = got.find(key);
    if (it == got.end()) {
      expect(false, key + ": not run");
      continue;
    }
    const Result& r = it->second;
    char buf[256];
    std::snprintf(buf, sizeof buf, "%-22s expected %2d %-10s %.3f   got %2d %-10s %.3f (%s)", key.c_str(), e.status,
                  e.id.c_str(), e.score, r.status, r.id.c_str(), r.score, palm_status_text(r.status));
    bool ok = r.status == e.status && r.id == e.id && std::fabs(r.score - e.score) <= 0.01f;
    if (ok && key.rfind("extract.", 0) == 0 && r.status == PALM_OK) {
      const std::string name = key.substr(8);
      std::vector<float> ref(PALM_FEATURE_DIM);
      std::ifstream(fs::path(data) / "features" / (name + ".f32"), std::ios::binary)
          .read(reinterpret_cast<char*>(ref.data()), PALM_FEATURE_DIM * sizeof(float));
      const float c = cosine(ref, feat[name]);
      ok = c >= 0.999f;
      std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), "  feature cos %.5f", c);
    }
    expect(ok, buf);
  }
  expect(steps == static_cast<int>(got.size()), "all " + std::to_string(got.size()) + " steps recorded");
  std::printf("%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
