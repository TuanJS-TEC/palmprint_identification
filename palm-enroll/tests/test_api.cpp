// C API (palm_api.h) end to end on full frames of the capture log.
// usage: test_api CONFIG.yaml LABELS.csv FRAMES_DIR
// FRAMES_DIR/<filename>_full.jpg are mirror images of the camera view
// (output/original). Exits 77 (skipped) when the models or frames are missing.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../tools/labels.h"
#include "check.h"
#include "config.h"
#include "gallery.h"
#include "palm_api.h"

namespace fs = std::filesystem;

namespace {

PalmFrame frameOf(const cv::Mat& m, int format, int mirrored = 0, int rotation = 0) {
  PalmFrame f{};
  f.data = m.data;
  f.size = m.step * m.rows;
  f.width = m.cols;
  f.height = m.rows;
  f.stride = static_cast<int>(m.step);
  f.format = format;
  f.rotation = rotation;
  f.mirrored = mirrored;
  return f;
}

// BGR -> NV21 bytes (Y plane, then V/U interleaved)
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

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return static_cast<float>(s);
}

struct Probe {
  std::string path;
  std::vector<float> feat;
};

}  // namespace

int main(int argc, char** argv) {
  // ---- arguments and NULL handling: no model needed
  for (int s : {0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 20, 21, 30, 31})
    CHECK(std::strcmp(palm_status_text(s), "unknown status") != 0);
  CHECK(std::strcmp(palm_status_text(999), "unknown status") == 0);
  palm_release(nullptr);
  char id[PALM_ID_MAX];
  float feat[PALM_FEATURE_DIM] = {};
  float score = -1;
  int same = -1;
  PalmFrame none{};
  CHECK(palm_extract(nullptr, &none, feat, PALM_FEATURE_DIM) == PALM_ERR_ARGUMENT);
  CHECK(palm_compare(nullptr, feat, feat, PALM_FEATURE_DIM, &score, &same) == PALM_ERR_ARGUMENT);
  CHECK(score == 0.f && same == 0);
  CHECK(palm_register(nullptr, &none, 1, id, sizeof id, nullptr, nullptr) == PALM_ERR_ARGUMENT);
  CHECK(palm_check_register(nullptr, &none, id, sizeof id, nullptr) == PALM_ERR_ARGUMENT);
  CHECK(palm_last_error(nullptr) != nullptr);
  PalmHandle h = reinterpret_cast<PalmHandle>(1);
  CHECK(palm_init(nullptr, &h) == PALM_ERR_ARGUMENT && h == nullptr);
  PalmInitOptions bare{};
  bare.embed_onnx = "x.onnx";  // config_path NULL: all four paths needed
  h = reinterpret_cast<PalmHandle>(1);
  CHECK(palm_init(&bare, &h) == PALM_ERR_INIT && h == nullptr);
  CHECK(std::strstr(palm_last_error(nullptr), "missing") != nullptr);

  if (argc < 4) {
    std::printf("usage: test_api CONFIG.yaml LABELS.csv FRAMES_DIR\n");
    return 1;
  }
  const palm::Config cfg = palm::loadConfig(argv[1]);
  for (const auto& p : {cfg.model.onnxPath, cfg.hand.palmOnnx, cfg.hand.landmarkOnnx, std::string(argv[2]),
                        std::string(argv[3])})
    if (!fs::exists(p)) {
      std::printf("skip: %s missing\n", p.c_str());
      return g_failures ? finish("test_api") : 77;
    }

  const fs::path tmp = fs::temp_directory_path() / "palm_test_api";
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  const std::string galleryPath = (tmp / "gallery.bin").string();
  PalmInitOptions opt{};
  opt.config_path = argv[1];
  opt.gallery_path = galleryPath.c_str();
  h = nullptr;
  CHECK_MSG(palm_init(&opt, &h) == PALM_OK, "%s", palm_last_error(nullptr));
  if (!h) return finish("test_api");

  // frames of the capture log: mirror images, right hand = cfg.rightLabel
  std::map<std::string, std::vector<Row>> right, left;
  for (const auto& r : readLabels(argv[2])) (r.hand == cfg.rightLabel ? right : left)[r.person].push_back(r);
  for (auto* m : {&right, &left})
    for (auto& [p, rows] : *m)
      std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ts < b.ts; });
  const auto framePath = [&](const Row& r) { return std::string(argv[3]) + "/" + r.file + "_full.jpg"; };

  // ---- frame formats: same pixels -> same feature
  const cv::Mat logFrame = cv::imread(framePath(right.begin()->second.front()), cv::IMREAD_COLOR);
  CHECK(!logFrame.empty());
  cv::Mat cam;  // the real (non-mirrored) camera view
  cv::flip(logFrame, cam, 1);
  std::vector<float> ref(PALM_FEATURE_DIM), f(PALM_FEATURE_DIM);
  PalmFrame fr = frameOf(logFrame, PALM_FMT_BGR, 1);
  CHECK_MSG(palm_extract(h, &fr, ref.data(), PALM_FEATURE_DIM) == PALM_OK, "%s", palm_last_error(h));
  CHECK(std::fabs(cosine(ref, ref) - 1.f) < 1e-4f);
  fr = frameOf(cam, PALM_FMT_BGR);
  CHECK(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM) == PALM_OK && cosine(ref, f) > 0.9999f);
  cv::Mat rgba;
  cv::cvtColor(cam, rgba, cv::COLOR_BGR2RGBA);
  fr = frameOf(rgba, PALM_FMT_RGBA);
  CHECK(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM) == PALM_OK && cosine(ref, f) > 0.9999f);
  cv::Mat sensor;  // buffer that needs a 90 degree clockwise turn
  cv::rotate(cam, sensor, cv::ROTATE_90_COUNTERCLOCKWISE);
  fr = frameOf(sensor, PALM_FMT_BGR, 0, 90);
  CHECK(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM) == PALM_OK && cosine(ref, f) > 0.9999f);
  cv::Mat padded(cam.rows, cam.cols + 32, CV_8UC3, cv::Scalar::all(0));
  cam.copyTo(padded(cv::Rect(0, 0, cam.cols, cam.rows)));
  fr = frameOf(padded(cv::Rect(0, 0, cam.cols, cam.rows)), PALM_FMT_BGR);  // stride > width * 3
  fr.size = padded.step * padded.rows;
  CHECK(fr.stride > cam.cols * 3);
  CHECK(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM) == PALM_OK && cosine(ref, f) > 0.9999f);
  const std::vector<unsigned char> nv21 = toNv21(cam);
  PalmFrame yuv{nv21.data(), nv21.size(), cam.cols, cam.rows, 0, PALM_FMT_NV21, 0, 0};
  CHECK(palm_extract(h, &yuv, f.data(), PALM_FEATURE_DIM) == PALM_OK);
  CHECK_MSG(cosine(ref, f) > 0.98f, "NV21 cos %.4f", cosine(ref, f));
  std::printf("NV21 vs BGR: cos %.4f\n", cosine(ref, f));
  // the caller's buffer is only read, also when it is turned or mirrored
  for (int rot : {0, 180}) {
    const cv::Mat before = logFrame.clone();
    fr = frameOf(logFrame, PALM_FMT_BGR, 1, rot);
    palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM);
    CHECK_MSG(cv::norm(before, logFrame, cv::NORM_INF) == 0, "rotation %d wrote into the frame", rot);
  }

  // ---- bad frames and buffers
  const auto argErr = [&](PalmFrame x) { return palm_extract(h, &x, f.data(), PALM_FEATURE_DIM) == PALM_ERR_ARGUMENT; };
  PalmFrame x = yuv;
  x.width -= 1;  // odd NV21 width
  CHECK(argErr(x));
  x = yuv;
  x.size -= 1;
  CHECK(argErr(x));
  x = frameOf(cam, PALM_FMT_BGR);
  x.stride = cam.cols * 3 - 1;
  CHECK(argErr(x));
  x = frameOf(cam, PALM_FMT_BGR);
  x.rotation = 45;
  CHECK(argErr(x));
  x.rotation = 0;
  x.format = 99;
  CHECK(argErr(x));
  x.format = PALM_FMT_BGR;
  x.data = nullptr;
  CHECK(argErr(x));
  x = frameOf(cam, PALM_FMT_BGR);
  x.width = 0;
  CHECK(argErr(x));
  CHECK(std::strlen(palm_last_error(h)) > 0);
  fr = frameOf(cam, PALM_FMT_BGR);
  CHECK(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM - 1) == PALM_ERR_ARGUMENT);
  CHECK(palm_extract(h, nullptr, f.data(), PALM_FEATURE_DIM) == PALM_ERR_ARGUMENT);
  CHECK(palm_extract(h, &fr, nullptr, PALM_FEATURE_DIM) == PALM_ERR_ARGUMENT);
  const cv::Mat black(480, 640, CV_8UC3, cv::Scalar::all(0));
  fr = frameOf(black, PALM_FMT_BGR);
  CHECK(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM) == PALM_ERR_NO_HAND);

  // ---- compare
  CHECK(palm_compare(h, ref.data(), ref.data(), PALM_FEATURE_DIM, &score, &same) == PALM_OK);
  CHECK(std::fabs(score - 1.f) < 1e-4f && same == 1);
  std::vector<float> zero(PALM_FEATURE_DIM, 0.f), nan = ref;
  nan[7] = std::numeric_limits<float>::quiet_NaN();
  CHECK(palm_compare(h, ref.data(), zero.data(), PALM_FEATURE_DIM, &score, &same) == PALM_ERR_ARGUMENT);
  CHECK(score == 0.f && same == 0);
  CHECK(palm_compare(h, nan.data(), ref.data(), PALM_FEATURE_DIM, &score, &same) == PALM_ERR_ARGUMENT);
  CHECK(palm_compare(h, ref.data(), ref.data(), PALM_FEATURE_DIM - 1, &score, &same) == PALM_ERR_ARGUMENT);
  CHECK(palm_compare(h, nullptr, ref.data(), PALM_FEATURE_DIM, &score, &same) == PALM_ERR_ARGUMENT);
  CHECK(palm_compare(h, ref.data(), ref.data(), PALM_FEATURE_DIM, nullptr, nullptr) == PALM_OK);

  // ---- register / check arguments, empty gallery
  fr = frameOf(cam, PALM_FMT_BGR);
  CHECK(palm_check_register(h, &fr, id, sizeof id, &score) == PALM_ERR_GALLERY_EMPTY);
  CHECK(palm_check_register(h, &fr, id, PALM_ID_MAX - 1, &score) == PALM_ERR_ARGUMENT);
  CHECK(palm_check_register(h, &fr, nullptr, PALM_ID_MAX, &score) == PALM_ERR_ARGUMENT);
  CHECK(palm_check_register(h, nullptr, id, sizeof id, &score) == PALM_ERR_ARGUMENT);
  CHECK(palm_register(h, &fr, 0, id, sizeof id, &score, nullptr) == PALM_ERR_ARGUMENT);
  CHECK(palm_register(h, &fr, PALM_MAX_FRAMES + 1, id, sizeof id, &score, nullptr) == PALM_ERR_ARGUMENT);
  CHECK(palm_register(h, nullptr, 1, id, sizeof id, &score, nullptr) == PALM_ERR_ARGUMENT);
  CHECK(palm_register(h, &fr, 1, id, 8, &score, nullptr) == PALM_ERR_ARGUMENT);
  CHECK(fs::exists(galleryPath) == false);

  // ---- usable right-hand frames of three people (some touch the frame edge)
  std::vector<std::vector<Probe>> people;
  for (const auto& [person, rows] : right) {
    std::vector<Probe> usable;
    for (const auto& r : rows) {
      const cv::Mat img = cv::imread(framePath(r), cv::IMREAD_COLOR);
      if (img.empty()) continue;
      fr = frameOf(img, PALM_FMT_BGR, 1);
      Probe p{framePath(r), std::vector<float>(PALM_FEATURE_DIM)};
      if (palm_extract(h, &fr, p.feat.data(), PALM_FEATURE_DIM) == PALM_OK) usable.push_back(std::move(p));
    }
    if (usable.size() >= 16) people.push_back(std::move(usable));
    if (people.size() == 3) break;
  }
  CHECK_MSG(people.size() == 3, "only %zu people with 16 usable frames", people.size());
  if (people.size() < 3) return finish("test_api");
  const auto registerPaths = [&](const std::vector<std::string>& paths, int* status, float* sc) {
    std::vector<cv::Mat> imgs;
    std::vector<PalmFrame> frames;
    for (const auto& p : paths) imgs.push_back(cv::imread(p, cv::IMREAD_COLOR));
    for (const auto& m : imgs) frames.push_back(frameOf(m, PALM_FMT_BGR, 1));
    return palm_register(h, frames.data(), static_cast<int>(frames.size()), id, sizeof id, sc, status);
  };
  const auto pathsOf = [](const std::vector<Probe>& v, size_t from, size_t to) {
    std::vector<std::string> out;
    for (size_t k = from; k < to && k < v.size(); ++k) out.push_back(v[k].path);
    return out;
  };
  const auto check = [&](const std::string& path) {
    const cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    PalmFrame q = frameOf(img, PALM_FMT_BGR, 1);
    return palm_check_register(h, &q, id, sizeof id, &score);
  };
  const auto& A = people[0];
  const auto& B = people[1];
  const auto& C = people[2];

  // ---- Register
  int status[6] = {-1, -1, -1, -1, -1, -1};
  CHECK_MSG(registerPaths(pathsOf(A, 0, 6), status, &score) == PALM_OK, "%s", palm_last_error(h));
  CHECK(std::string(id) == "user_0001" && score == 0.f);
  CHECK(std::all_of(status, status + 6, [](int s) { return s == PALM_OK; }));
  CHECK(fs::exists(galleryPath));
  const int again = registerPaths(pathsOf(A, 6, 12), nullptr, &score);
  CHECK_MSG(again == PALM_ERR_ALREADY_REGISTERED && std::string(id) == "user_0001", "%d %s cos %.3f", again, id,
            score);
  CHECK(score > cfg.dupCosMax);
  CHECK_MSG(registerPaths(pathsOf(B, 0, 6), nullptr, &score) == PALM_OK, "%s", palm_last_error(h));
  CHECK(std::string(id) == "user_0002");
  CHECK(registerPaths(pathsOf(A, 0, 2), nullptr, &score) == PALM_ERR_TOO_FEW_SAMPLES && id[0] == '\0');
  std::vector<cv::Mat> blacks(3, black);
  std::vector<PalmFrame> bf;
  for (const auto& m : blacks) bf.push_back(frameOf(m, PALM_FMT_BGR));
  CHECK(palm_register(h, bf.data(), 3, id, sizeof id, &score, nullptr) == PALM_ERR_NO_HAND);
  // left hands of a person never register
  const auto& lrows = left.begin()->second;
  std::vector<std::string> lpaths;
  for (size_t k = 0; k < lrows.size() && lpaths.size() < 6; ++k) lpaths.push_back(framePath(lrows[k]));
  int lstatus[6] = {};
  const int ls = registerPaths(lpaths, lstatus, &score);
  CHECK_MSG(ls != PALM_OK, "left hand registered as %s", id);
  int leftOrBack = 0;
  for (int s : lstatus) leftOrBack += s == PALM_ERR_LEFT_HAND || s == PALM_ERR_BACK_OF_HAND;
  CHECK_MSG(leftOrBack >= 4, "only %d/6 left-hand frames flagged", leftOrBack);

  // ---- Check_Register
  int hit = 0, wrong = 0, n = 0;
  for (size_t k = 6; k < A.size(); ++k, ++n) {
    const int s = check(A[k].path);
    if (s == PALM_OK) (std::string(id) == "user_0001" ? hit : wrong) += 1;
    else CHECK(id[0] == '\0');
  }
  std::printf("check A: %d/%d accepted, %d wrong id\n", hit, n, wrong);
  CHECK(wrong == 0 && hit >= 0.7 * n);
  int falseAccept = 0;
  for (const auto& p : C) falseAccept += check(p.path) == PALM_OK;
  CHECK_MSG(falseAccept == 0, "%d frames of an unregistered person accepted", falseAccept);
  CHECK(check(C[0].path) == PALM_ERR_NOT_RECOGNIZED && id[0] == '\0' && score < cfg.acceptCosMin);

  // ---- persistence, id counter, other writers
  palm_release(h);
  h = nullptr;
  CHECK(palm_init(&opt, &h) == PALM_OK);
  CHECK(check(A[8].path) == PALM_OK || check(A[9].path) == PALM_OK);
  CHECK_MSG(registerPaths(pathsOf(C, 0, 6), nullptr, &score) == PALM_OK, "%s", palm_last_error(h));
  CHECK(std::string(id) == "user_0003");
  {  // another process removes user_0003: its id is not handed out again
    palm::Gallery g(cfg.modelVersion);
    g.load(galleryPath);
    CHECK(g.remove("user_0003"));
    g.save(galleryPath);
  }
  CHECK(check(C[8].path) != PALM_OK);  // reloaded: C is gone
  CHECK(registerPaths(pathsOf(C, 0, 6), nullptr, &score) == PALM_OK && std::string(id) == "user_0004");

  // ---- unreadable gallery file: logins use the copy in memory, register refuses to overwrite it
  std::ofstream(galleryPath, std::ios::binary | std::ios::trunc) << "not a gallery";
  int s = PALM_ERR_INTERNAL;
  for (size_t k = 6; k < A.size() && s != PALM_OK; ++k) s = check(A[k].path);
  CHECK(s == PALM_OK && std::string(id) == "user_0001");
  CHECK(std::strstr(palm_last_error(h), "cannot read the gallery") != nullptr);
  CHECK(registerPaths(pathsOf(B, 6, 12), nullptr, &score) == PALM_ERR_IO);
  palm_release(h);

  // ---- save failure: nothing enrolled, error reported
  const std::string noDir = (tmp / "missing_dir" / "gallery.bin").string();
  opt.gallery_path = noDir.c_str();
  h = nullptr;
  CHECK(palm_init(&opt, &h) == PALM_OK);
  CHECK(registerPaths(pathsOf(A, 0, 6), nullptr, &score) == PALM_ERR_IO && id[0] == '\0');
  CHECK(std::strstr(palm_last_error(h), "cannot save") != nullptr);
  CHECK(check(A[7].path) == PALM_ERR_GALLERY_EMPTY);
  palm_release(h);

  // ---- no config file: built-in thresholds, explicit paths
  PalmInitOptions direct{};
  const std::string g2 = (tmp / "g2.bin").string();
  direct.embed_onnx = cfg.model.onnxPath.c_str();
  direct.palm_onnx = cfg.hand.palmOnnx.c_str();
  direct.landmark_onnx = cfg.hand.landmarkOnnx.c_str();
  direct.gallery_path = g2.c_str();
  h = nullptr;
  CHECK_MSG(palm_init(&direct, &h) == PALM_OK, "%s", palm_last_error(nullptr));
  fr = frameOf(cam, PALM_FMT_BGR);
  CHECK(palm_extract(h, &fr, f.data(), PALM_FEATURE_DIM) == PALM_OK && cosine(ref, f) > 0.9999f);
  palm_release(h);

  fs::remove_all(tmp);
  return finish("test_api");
}
