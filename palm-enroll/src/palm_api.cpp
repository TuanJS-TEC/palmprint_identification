// C API over the palm pipeline (see palm_api.h): frame -> hand detector ->
// right-palm checks -> hand crop -> EE-PRnet feature -> gallery.
#include "palm_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "config.h"
#include "gallery.h"
#include "hand_crop.h"
#include "hand_detector.h"
#include "palm_model.h"

namespace fs = std::filesystem;

namespace {

// Gallery file identity, to notice when another process (palm_gui, palm_cli)
// rewrote it. Size and the SQLite file change counter as well as mtime: two
// writes can share an mtime tick, and the counter moves with every commit.
struct FileStamp {
  bool exists = false;
  fs::file_time_type time{};
  std::uintmax_t size = 0;
  std::uint32_t changeCounter = 0;
  bool operator==(const FileStamp& o) const {
    return exists == o.exists && time == o.time && size == o.size && changeCounter == o.changeCounter;
  }
};

FileStamp stampOf(const std::string& path) {
  std::error_code ec;
  FileStamp s;
  s.time = fs::last_write_time(path, ec);
  if (ec) return {};
  s.size = fs::file_size(path, ec);
  if (ec) return {};
  s.exists = true;
  // header bytes 24..27, big-endian; absent in a file that is not a database
  std::ifstream f(path, std::ios::binary);
  unsigned char h[28];
  if (f.seekg(0) && f.read(reinterpret_cast<char*>(h), sizeof h))
    s.changeCounter = (std::uint32_t{h[24]} << 24) | (std::uint32_t{h[25]} << 16) | (std::uint32_t{h[26]} << 8) | h[27];
  return s;
}

// Values of config.yaml (measured with palm_eval, 2026-10-05) for callers
// without a config file; keep in sync.
palm::Config measuredConfig() {
  palm::Config c;
  c.model.threads = 4;
  c.tmpl.minValid = 3;
  c.tmpl.outlierCosMin = 0.5f;
  c.dupCosMax = 0.82f;
  c.acceptCosMin = 0.82f;
  c.capture.cropScale = 1.17f;
  c.capture.minCropFrac = 0.33f;
  c.capture.maxMotion = 0.02f;
  c.capture.blurMin = 100.f;
  return c;
}

thread_local std::string g_initError;  // palm_last_error(NULL)

void setInitError(const char* msg) noexcept {
  try {
    g_initError = msg;
  } catch (...) {
  }
}

}  // namespace

struct PalmContext {
  palm::Config cfg;
  palm::PalmModel model;
  palm::HandDetector det;
  palm::Gallery gallery;
  FileStamp stamp;
  std::mutex m;
  std::string lastError;

  explicit PalmContext(const palm::Config& c)
      : cfg(c),
        model(c.model),
        det(c.hand.palmOnnx, c.hand.landmarkOnnx, 2),
        gallery(c.modelVersion, model.featureDim()) {}
};

namespace {

using palm::HandState;
using palm::Vec;

std::string fmt3(float v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3f", v);
  return b;
}

int fail(PalmContext& c, int status, std::string msg) {
  c.lastError = std::move(msg);
  return status;
}

// Locks the handle and turns every exception into PALM_ERR_INTERNAL: nothing
// may unwind through the C boundary.
template <typename F>
int guarded(PalmContext* c, F&& body) noexcept {
  try {
    std::lock_guard<std::mutex> lock(c->m);
    c->lastError.clear();
    try {
      return body(*c);
    } catch (const std::exception& e) {
      c->lastError = e.what();
    } catch (...) {
      c->lastError = "unknown exception";
    }
    return PALM_ERR_INTERNAL;
  } catch (...) {
    return PALM_ERR_INTERNAL;
  }
}

// Copies id with its '\0', never truncated: false when it does not fit.
bool copyId(const std::string& id, char* out, int cap) {
  if (cap < 1 || id.size() + 1 > static_cast<size_t>(cap)) return false;
  std::snprintf(out, static_cast<size_t>(cap), "%s", id.c_str());
  return true;
}

// Upright, non-mirrored BGR view or copy of the caller's buffer.
int toBgr(const PalmFrame& f, cv::Mat& bgr, std::string& err) {
  constexpr int kMaxSide = 16384;
  const auto bad = [&](std::string msg) -> int {
    err = std::move(msg);
    return PALM_ERR_ARGUMENT;
  };
  if (!f.data) return bad("frame.data is NULL");
  if (f.width < 1 || f.height < 1 || f.width > kMaxSide || f.height > kMaxSide)
    return bad("frame width/height must be 1..16384");
  int bpp = 0;
  switch (f.format) {
    case PALM_FMT_BGR:
    case PALM_FMT_RGB: bpp = 3; break;
    case PALM_FMT_BGRA:
    case PALM_FMT_RGBA: bpp = 4; break;
    case PALM_FMT_NV21: bpp = 1; break;
    default: return bad("unknown frame.format " + std::to_string(f.format));
  }
  const bool nv21 = f.format == PALM_FMT_NV21;
  if (nv21 && (f.width % 2 || f.height % 2)) return bad("NV21 needs an even width and height");
  const size_t row = static_cast<size_t>(f.width) * bpp;
  if (f.stride < 0 || (f.stride > 0 && static_cast<size_t>(f.stride) < row))
    return bad("frame.stride is smaller than a row (" + std::to_string(row) + " bytes)");
  const size_t step = f.stride > 0 ? static_cast<size_t>(f.stride) : row;
  const size_t rows = static_cast<size_t>(f.height) + (nv21 ? f.height / 2 : 0);
  const size_t need = step * (rows - 1) + row;  // the last row may stop short of the stride
  if (f.size < need) return bad("frame.size too small: need " + std::to_string(need) + " bytes");
  if (f.rotation != 0 && f.rotation != 90 && f.rotation != 180 && f.rotation != 270)
    return bad("frame.rotation must be 0, 90, 180 or 270");

  auto* p = const_cast<unsigned char*>(f.data);  // only read: cv::Mat has no const view
  const int w = f.width, h = f.height;
  switch (f.format) {
    case PALM_FMT_BGR: bgr = cv::Mat(h, w, CV_8UC3, p, step); break;
    case PALM_FMT_RGB: cv::cvtColor(cv::Mat(h, w, CV_8UC3, p, step), bgr, cv::COLOR_RGB2BGR); break;
    case PALM_FMT_BGRA: cv::cvtColor(cv::Mat(h, w, CV_8UC4, p, step), bgr, cv::COLOR_BGRA2BGR); break;
    case PALM_FMT_RGBA: cv::cvtColor(cv::Mat(h, w, CV_8UC4, p, step), bgr, cv::COLOR_RGBA2BGR); break;
    case PALM_FMT_NV21:
      cv::cvtColorTwoPlane(cv::Mat(h, w, CV_8UC1, p, step), cv::Mat(h / 2, w / 2, CV_8UC2, p + h * step, step), bgr,
                           cv::COLOR_YUV2BGR_NV21);
      break;
  }
  // never in place: for PALM_FMT_BGR, bgr is still the caller's (const) buffer
  if (f.rotation) {
    cv::Mat r;
    cv::rotate(bgr, r, f.rotation == 90 ? cv::ROTATE_90_CLOCKWISE
                       : f.rotation == 180 ? cv::ROTATE_180 : cv::ROTATE_90_COUNTERCLOCKWISE);
    bgr = r;
  }
  if (f.mirrored) {
    cv::Mat m;
    cv::flip(bgr, m, 1);
    bgr = m;
  }
  return PALM_OK;
}

int stateStatus(HandState s) {
  switch (s) {
    case HandState::Ok: return PALM_OK;
    case HandState::NoHand: return PALM_ERR_NO_HAND;
    case HandState::LeftHand: return PALM_ERR_LEFT_HAND;
    case HandState::BackOfHand: return PALM_ERR_BACK_OF_HAND;
    case HandState::TooSmall: return PALM_ERR_HAND_TOO_SMALL;
    case HandState::OutOfFrame: return PALM_ERR_HAND_OUT_OF_FRAME;
    case HandState::Moving:  // not raised for a single frame
    case HandState::Blurry: return PALM_ERR_BLURRY;
  }
  return PALM_ERR_INTERNAL;
}

// The live path of palm_gui on one frame: detect, check, crop, embed.
int embedFrame(PalmContext& c, const PalmFrame& f, Vec& feat) {
  cv::Mat bgr;
  if (const int s = toBgr(f, bgr, c.lastError)) return s;
  const palm::HandLandmarks h = c.det.detect(bgr);
  const palm::HandCheck chk = palm::checkHand(bgr, h, h, c.cfg.capture);  // single frame: no motion
  if (const int s = stateStatus(chk.state)) return fail(c, s, palm_status_text(s));
  feat = palm::l2normalize(c.model.run(bgr(chk.box), false).feature);  // upright, not mirrored
  return PALM_OK;
}

// Reloads the gallery when its file changed. On a read error the gallery in
// memory is kept, lastError says why and false is returned.
bool syncGallery(PalmContext& c) {
  const FileStamp s = stampOf(c.cfg.galleryPath);
  if (s == c.stamp) return true;
  try {
    c.gallery.load(c.cfg.galleryPath);
  } catch (const std::exception& e) {
    c.lastError = std::string("cannot read the gallery, using the copy in memory: ") + e.what();
    return false;
  }
  c.stamp = s;
  return true;
}

// Copy of v[0..dim) scaled to unit length; false for a zero or non-finite vector.
bool normalised(const float* v, int dim, Vec& out) {
  double n2 = 0;
  for (int i = 0; i < dim; ++i) {
    if (!std::isfinite(v[i])) return false;
    n2 += static_cast<double>(v[i]) * v[i];
  }
  const double n = std::sqrt(n2);
  if (!(n > 1e-12) || !std::isfinite(n)) return false;
  out.resize(dim);
  for (int i = 0; i < dim; ++i) out[i] = static_cast<float>(v[i] / n);
  return true;
}

}  // namespace

extern "C" {

int palm_init(const PalmInitOptions* opt, PalmHandle* out) {
  try {
    g_initError.clear();
    if (out) *out = nullptr;
    if (!opt || !out) {
      setInitError("palm_init: opt and out must not be NULL");
      return PALM_ERR_ARGUMENT;
    }
    palm::Config cfg;
    if (opt->config_path) {
      cfg = palm::loadConfig(opt->config_path);
    } else {
      if (!opt->embed_onnx || !opt->palm_onnx || !opt->landmark_onnx || !opt->gallery_path) {
        setInitError("palm_init: missing model/gallery paths when config_path is NULL");
        return PALM_ERR_INIT;
      }
      cfg = measuredConfig();
    }
    if (opt->embed_onnx) cfg.model.onnxPath = opt->embed_onnx;
    if (opt->palm_onnx) cfg.hand.palmOnnx = opt->palm_onnx;
    if (opt->landmark_onnx) cfg.hand.landmarkOnnx = opt->landmark_onnx;
    if (opt->gallery_path) cfg.galleryPath = opt->gallery_path;
    if (opt->threads > 0) cfg.model.threads = opt->threads;

    auto c = std::make_unique<PalmContext>(cfg);
    if (c->model.featureDim() != PALM_FEATURE_DIM) {
      setInitError(("palm_init: the model gives " + std::to_string(c->model.featureDim()) +
                    "-d features, expected PALM_FEATURE_DIM").c_str());
      return PALM_ERR_INIT;
    }
    c->stamp = stampOf(cfg.galleryPath);
    c->gallery.load(cfg.galleryPath);
    *out = c.release();
    return PALM_OK;
  } catch (const std::exception& e) {
    setInitError(e.what());
  } catch (...) {
    setInitError("palm_init: unknown exception");
  }
  return PALM_ERR_INIT;
}

void palm_release(PalmHandle h) { delete h; }

int palm_extract(PalmHandle h, const PalmFrame* frame, float* feature, int feature_len) {
  if (!h) return PALM_ERR_ARGUMENT;
  return guarded(h, [&](PalmContext& c) -> int {
    if (!frame || !feature) return fail(c, PALM_ERR_ARGUMENT, "frame and feature must not be NULL");
    if (feature_len < PALM_FEATURE_DIM) return fail(c, PALM_ERR_ARGUMENT, "feature_len < PALM_FEATURE_DIM");
    Vec f;
    if (const int s = embedFrame(c, *frame, f)) return s;
    std::copy(f.begin(), f.end(), feature);
    return PALM_OK;
  });
}

int palm_compare(PalmHandle h, const float* a, const float* b, int dim, float* score, int* same) {
  if (score) *score = 0.f;
  if (same) *same = 0;
  if (!h) return PALM_ERR_ARGUMENT;
  return guarded(h, [&](PalmContext& c) -> int {
    if (!a || !b) return fail(c, PALM_ERR_ARGUMENT, "a and b must not be NULL");
    if (dim != PALM_FEATURE_DIM) return fail(c, PALM_ERR_ARGUMENT, "dim must be PALM_FEATURE_DIM");
    Vec va, vb;
    if (!normalised(a, dim, va) || !normalised(b, dim, vb))
      return fail(c, PALM_ERR_ARGUMENT, "a zero or non-finite feature vector");
    const float s = palm::dot(va, vb);
    if (score) *score = s;
    if (same) *same = s >= c.cfg.acceptCosMin;
    return PALM_OK;
  });
}

int palm_register(PalmHandle h, const PalmFrame* frames, int n, char* id_out, int id_max_len, float* score,
                  int* frame_status) {
  const bool idBuf = id_out && id_max_len >= PALM_ID_MAX;
  if (idBuf) id_out[0] = '\0';
  if (score) *score = 0.f;
  if (!h) return PALM_ERR_ARGUMENT;
  return guarded(h, [&](PalmContext& c) -> int {
    if (!idBuf) return fail(c, PALM_ERR_ARGUMENT, "id_out must hold at least PALM_ID_MAX bytes");
    if (!frames || n < 1 || n > PALM_MAX_FRAMES) return fail(c, PALM_ERR_ARGUMENT, "need 1..PALM_MAX_FRAMES frames");

    std::vector<Vec> feats;
    std::map<int, int> rejected;  // status -> frames
    for (int i = 0; i < n; ++i) {
      Vec f;
      const int s = embedFrame(c, frames[i], f);
      if (frame_status) frame_status[i] = s;
      if (s == PALM_ERR_ARGUMENT) return fail(c, s, "frame " + std::to_string(i) + ": " + c.lastError);
      if (s == PALM_OK) feats.push_back(std::move(f));
      else ++rejected[s];
    }
    const int used = static_cast<int>(feats.size());
    if (used == 0) {  // the most frequent reason tells the user what to change
      const auto top = std::max_element(rejected.begin(), rejected.end(),
                                        [](const auto& x, const auto& y) { return x.second < y.second; });
      return fail(c, top->first, std::string("no usable frame, mostly: ") + palm_status_text(top->first));
    }
    if (used < c.cfg.tmpl.minValid)
      return fail(c, PALM_ERR_TOO_FEW_SAMPLES, std::to_string(used) + " of " + std::to_string(n) +
                                                   " frames usable, need " + std::to_string(c.cfg.tmpl.minValid));
    const palm::TemplateResult t = palm::buildTemplate(feats, c.cfg.tmpl);
    if (!t.ok)
      return fail(c, PALM_ERR_TOO_FEW_SAMPLES, std::to_string(t.kept.size()) + " of " + std::to_string(used) +
                                                   " usable frames consistent, need " +
                                                   std::to_string(c.cfg.tmpl.minValid));

    if (!syncGallery(c)) return PALM_ERR_IO;  // never overwrite a gallery file that cannot be read
    const std::vector<Vec> samples = palm::keptSamples(feats, t);
    const auto best = c.gallery.search(samples, 1);  // closest pair of samples
    if (!best.empty() && score) *score = best[0].score;
    if (!best.empty() && best[0].score > c.cfg.dupCosMax) {
      if (!copyId(best[0].id, id_out, id_max_len))
        return fail(c, PALM_ERR_INTERNAL, "the enrolled id is longer than PALM_ID_MAX - 1 bytes");
      return fail(c, PALM_ERR_ALREADY_REGISTERED, "the palm matches an enrolled user (cos " + fmt3(best[0].score) +
                                                      " > dup_cos_max " + fmt3(c.cfg.dupCosMax) + ")");
    }

    const std::string id = c.gallery.nextId();
    if (c.gallery.contains(id)) return fail(c, PALM_ERR_INTERNAL, "generated id already enrolled: " + id);
    c.gallery.upsert({id, samples});
    try {
      c.gallery.save(c.cfg.galleryPath);
    } catch (const std::exception& e) {
      c.gallery.remove(id);  // its number stays used, like a database auto-increment
      return fail(c, PALM_ERR_IO, std::string("cannot save the gallery: ") + e.what());
    }
    c.stamp = stampOf(c.cfg.galleryPath);
    if (!copyId(id, id_out, id_max_len)) return fail(c, PALM_ERR_INTERNAL, "generated id too long");
    return PALM_OK;
  });
}

int palm_check_register(PalmHandle h, const PalmFrame* frame, char* id_out, int id_max_len, float* score) {
  const bool idBuf = id_out && id_max_len >= PALM_ID_MAX;
  if (idBuf) id_out[0] = '\0';
  if (score) *score = 0.f;
  if (!h) return PALM_ERR_ARGUMENT;
  return guarded(h, [&](PalmContext& c) -> int {
    if (!idBuf) return fail(c, PALM_ERR_ARGUMENT, "id_out must hold at least PALM_ID_MAX bytes");
    if (!frame) return fail(c, PALM_ERR_ARGUMENT, "frame must not be NULL");
    Vec f;
    if (const int s = embedFrame(c, *frame, f)) return s;
    syncGallery(c);  // on a read error the copy in memory still serves logins
    if (c.gallery.entries().empty()) return fail(c, PALM_ERR_GALLERY_EMPTY, "no user enrolled");
    const auto best = c.gallery.search(f, 1);
    if (score) *score = best[0].score;
    if (best[0].score < c.cfg.acceptCosMin)  // the nearest id is not disclosed
      return fail(c, PALM_ERR_NOT_RECOGNIZED, "best score " + fmt3(best[0].score) + " < accept_cos_min " +
                                                      fmt3(c.cfg.acceptCosMin));
    if (!copyId(best[0].id, id_out, id_max_len))
      return fail(c, PALM_ERR_INTERNAL, "the enrolled id is longer than PALM_ID_MAX - 1 bytes");
    return PALM_OK;
  });
}

const char* palm_status_text(int status) {
  switch (status) {
    case PALM_OK: return "ok";
    case PALM_ERR_ARGUMENT: return "invalid argument";
    case PALM_ERR_INIT: return "initialisation failed";
    case PALM_ERR_IO: return "gallery read/write error";
    case PALM_ERR_INTERNAL: return "internal error";
    case PALM_ERR_NO_HAND: return "no hand in the frame";
    case PALM_ERR_LEFT_HAND: return "left hand: use the right hand";
    case PALM_ERR_BACK_OF_HAND: return "back of the hand: show the right palm";
    case PALM_ERR_HAND_TOO_SMALL: return "hand too small: move it closer";
    case PALM_ERR_HAND_OUT_OF_FRAME: return "hand not fully in the frame";
    case PALM_ERR_BLURRY: return "blurry frame: hold the hand still";
    case PALM_ERR_TOO_FEW_SAMPLES: return "too few usable frames";
    case PALM_ERR_ALREADY_REGISTERED: return "palm already registered";
    case PALM_ERR_GALLERY_EMPTY: return "no user registered";
    case PALM_ERR_NOT_RECOGNIZED: return "palm not recognised";
  }
  return "unknown status";
}

const char* palm_last_error(PalmHandle h) { return h ? h->lastError.c_str() : g_initError.c_str(); }

}  // extern "C"
