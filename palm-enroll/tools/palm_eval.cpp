// palm_eval: run the C++ pipeline on a labelled folder of hand crops and
// measure what the thresholds in config.yaml should be:
//   - sample vs sample: genuine / impostor cosine distributions, EER
//   - enrollment protocol: first --enroll samples of each palm (by timestamp)
//     -> template, the rest are probes: rank-1, EER, FRR at fixed FAR
//   - per-image timing
//
// As in enrollment, only the physical right hand is used (rows whose
// handedness is hand.right_label): one identity per person.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "args.h"
#include "config.h"
#include "gallery.h"
#include "hand_crop.h"
#include "hand_detector.h"
#include "labels.h"
#include "palm_model.h"

using namespace palm;

namespace {

const char* kUsage = R"(usage: palm_eval --labels LABELS.csv (--images DIR | --frames DIR [--frames-mirrored])
                 [--config config.yaml] [--flip] [--enroll 5] [--limit N]
                 [--scores scores.csv] [--roi-sheet rois.jpg]
LABELS.csv: ';' or ',' separated, columns filename, timestamp, handedness, person_id;
            rows of the left hand are ignored
--images:  hand crops, DIR/<filename>.jpg
--frames:  full camera frames, DIR/<filename>_full.jpg, run through the live path
           of palm_gui (hand detector, checks, crop); --frames-mirrored if the
           frames are mirror images of the camera view (output/original)
--flip:    mirror the hand crops first (on top of preprocess.mirror_input)
)";

// ---------------------------------------------------------------- statistics
double pct(std::vector<float> v, double p) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const double x = p / 100.0 * (v.size() - 1);
  const size_t i = static_cast<size_t>(x);
  return i + 1 < v.size() ? v[i] + (x - i) * (v[i + 1] - v[i]) : v[i];
}

void describe(const char* name, const std::vector<float>& v) {
  double m = 0, s = 0;
  for (float x : v) m += x;
  m /= std::max<size_t>(1, v.size());
  for (float x : v) s += (x - m) * (x - m);
  s = std::sqrt(s / std::max<size_t>(1, v.size()));
  std::printf("  %-26s n=%-7zu mean %.3f  std %.3f  p1 %.3f  p50 %.3f  p99 %.3f\n", name, v.size(), m, s,
              pct(v, 1), pct(v, 50), pct(v, 99));
}

struct Eer {
  double eer = NAN;
  float thr = NAN;
};

// accept rule: score >= threshold
Eer computeEer(std::vector<float> gen, std::vector<float> imp) {
  Eer best;
  if (gen.empty() || imp.empty()) return best;
  std::sort(gen.begin(), gen.end());
  std::sort(imp.begin(), imp.end());
  std::vector<float> cand(gen);
  cand.insert(cand.end(), imp.begin(), imp.end());
  std::sort(cand.begin(), cand.end());
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
  double bestGap = 2;
  for (float t : cand) {
    const double frr = double(std::lower_bound(gen.begin(), gen.end(), t) - gen.begin()) / gen.size();
    const double far = double(imp.end() - std::lower_bound(imp.begin(), imp.end(), t)) / imp.size();
    if (std::fabs(far - frr) < bestGap) bestGap = std::fabs(far - frr), best = {(far + frr) / 2, t};
  }
  return best;
}

struct Operating {
  float thr = NAN;
  double far = NAN, frr = NAN;
};

// lowest threshold with FAR <= target
Operating atFar(std::vector<float> gen, std::vector<float> imp, double target) {
  Operating o;
  if (gen.empty() || imp.empty()) return o;
  std::sort(imp.begin(), imp.end(), std::greater<float>());
  const size_t k = static_cast<size_t>(std::floor(target * imp.size()));
  o.thr = k < imp.size() ? std::nextafter(imp[k], std::numeric_limits<float>::infinity())
                         : -std::numeric_limits<float>::infinity();
  o.far = double(std::count_if(imp.begin(), imp.end(), [&](float s) { return s >= o.thr; })) / imp.size();
  o.frr = double(std::count_if(gen.begin(), gen.end(), [&](float s) { return s < o.thr; })) / gen.size();
  return o;
}

void report(const char* title, const std::vector<float>& gen, const std::vector<float>& imp) {
  const Eer e = computeEer(gen, imp);
  std::printf("  %s: EER %.2f%% at cos %.3f\n", title, 100 * e.eer, e.thr);
  for (double far : {1e-2, 1e-3}) {
    if (imp.size() * far < 5) continue;  // too few impostor scores to estimate this FAR
    const Operating o = atFar(gen, imp, far);
    std::printf("    FAR <= %-6g: threshold %.3f  FAR %.4f%%  FRR %.2f%%\n", far, o.thr, 100 * o.far, 100 * o.frr);
  }
}

double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a(argc, argv, 1, {"flip", "frames-mirrored"});
    if (!a.has("labels") || a.has("images") == a.has("frames")) {
      std::cerr << kUsage;
      return 1;
    }
    const Config cfg = loadConfig(a.get("config", "config.yaml"));
    const bool flip = a.has("flip") || cfg.mirrorInput;
    const int enrollK = std::stoi(a.get("enroll", "5"));
    const size_t limit = std::stoul(a.get("limit", "0"));

    auto rows = readLabels(a.get("labels"));
    const size_t labelled = rows.size();
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const Row& r) { return !cfg.isRightHand(r.hand); }),
               rows.end());
    std::stable_sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) { return x.ts < y.ts; });
    if (limit) rows.resize(std::min(limit, rows.size()));

    PalmModel model(cfg.model);
    std::printf("model %s  threads %d  input224 %s  mirror %d\n", cfg.model.onnxPath.c_str(), cfg.model.threads,
                cfg.model.input224Original ? "original" : "resize224", flip);
    std::printf("right hand only (handedness '%s'): %zu of %zu labelled rows\n", cfg.rightLabel.c_str(), rows.size(),
                labelled);

    // ---- embeddings
    const bool fromFrames = a.has("frames");
    std::unique_ptr<HandDetector> det;
    CaptureLimits lim = cfg.capture;
    lim.blurMin = 0;  // measured below, not applied
    if (fromFrames) det = std::make_unique<HandDetector>(cfg.hand.palmOnnx, cfg.hand.landmarkOnnx, cfg.model.threads);
    std::map<HandState, int> rejected;
    std::vector<Row> used;
    std::vector<Vec> feats;
    std::vector<cv::Mat> rois;
    std::vector<float> tPre, tInf, sharp;
    for (const auto& r : rows) {
      cv::Mat bgr;
      if (fromFrames) {
        cv::Mat frame = cv::imread(a.get("frames") + "/" + r.file + "_full.jpg", cv::IMREAD_COLOR);
        if (!frame.empty()) {
          if (a.has("frames-mirrored")) cv::flip(frame, frame, 1);
          const HandLandmarks h = det->detect(frame);
          const HandCheck chk = checkHand(frame, h, h, lim);  // single image: no motion
          if (chk.state != HandState::Ok) {
            ++rejected[chk.state];
            continue;
          }
          bgr = frame(chk.box).clone();
        }
      } else {
        bgr = cv::imread(a.get("images") + "/" + r.file + ".jpg", cv::IMREAD_COLOR);
      }
      if (bgr.empty()) {
        std::fprintf(stderr, "missing image: %s\n", r.file.c_str());
        continue;
      }
      sharp.push_back(sharpness(bgr));
      const auto t0 = std::chrono::steady_clock::now();
      const ModelInputs in = model.preprocess(bgr, flip);
      const auto t1 = std::chrono::steady_clock::now();
      Embedding e = model.infer(in);
      const auto t2 = std::chrono::steady_clock::now();
      tPre.push_back(static_cast<float>(ms(t0, t1)));
      tInf.push_back(static_cast<float>(ms(t1, t2)));
      used.push_back(r);
      feats.push_back(l2normalize(e.feature));
      rois.push_back(e.roi);
      if (used.size() % 100 == 0) std::fprintf(stderr, "  %zu/%zu\n", used.size(), rows.size());
    }
    const size_t n = used.size();
    std::printf("images: %zu embedded (%zu rows)\n", n, rows.size());
    for (const auto& [s, k] : rejected) std::printf("  rejected by live checks: %-42s %d\n", stateMessage(s), k);
    std::printf("timing per image: preprocess p50 %.1f ms, inference p50 %.1f ms (p95 %.1f ms)\n", pct(tPre, 50),
                pct(tInf, 50), pct(tInf, 95));
    std::printf("crop sharpness (var of Laplacian): p1 %.1f  p5 %.1f  p50 %.1f\n", pct(sharp, 1), pct(sharp, 5),
                pct(sharp, 50));

    std::map<std::string, std::vector<int>> byPerson;  // person_id -> sample indices
    for (size_t i = 0; i < n; ++i) byPerson[used[i].person].push_back(static_cast<int>(i));
    std::printf("people: %zu\n", byPerson.size());

    // ---- sample vs sample
    std::vector<float> gen, imp;
    for (size_t i = 0; i < n; ++i)
      for (size_t j = i + 1; j < n; ++j) (used[i].person == used[j].person ? gen : imp).push_back(dot(feats[i], feats[j]));
    std::printf("\n[sample vs sample]\n");
    describe("genuine (same person)", gen);
    describe("impostor (other person)", imp);
    report("sample-sample", gen, imp);

    // ---- enrollment protocol
    Gallery gallery(cfg.modelVersion, model.featureDim());
    std::vector<std::pair<int, std::string>> probes;  // (sample index, true person)
    std::vector<float> medoidCos;
    int skipped = 0, failed = 0;
    for (const auto& [person, idx] : byPerson) {
      if (static_cast<int>(idx.size()) <= enrollK) {
        ++skipped;
        continue;
      }
      std::vector<Vec> enr;
      for (int k = 0; k < enrollK; ++k) enr.push_back(feats[idx[k]]);
      const TemplateResult t = buildTemplate(enr, cfg.tmpl);
      if (!t.ok) {
        ++failed;
        std::printf("  template failed for %s (%s): cos to medoid", person.c_str(), t.reason.c_str());
        for (float c : t.cosToMedoid) std::printf(" %.2f", c);
        std::printf("\n");
        continue;
      }
      for (size_t k = 0; k < t.cosToMedoid.size(); ++k)
        if (static_cast<int>(k) != t.medoid) medoidCos.push_back(t.cosToMedoid[k]);
      gallery.upsert({person, static_cast<int>(t.kept.size()), t.tmpl});
      for (size_t k = enrollK; k < idx.size(); ++k) probes.emplace_back(idx[k], person);
    }
    std::vector<float> tGen, tImp;
    int rank1 = 0, acceptOk = 0, acceptWrong = 0;
    for (const auto& [i, truth] : probes) {
      const auto all = gallery.search(feats[i], static_cast<int>(gallery.entries().size()));
      for (const auto& m : all) (m.id == truth ? tGen : tImp).push_back(m.score);
      rank1 += all[0].id == truth;
      if (all[0].score >= cfg.acceptCosMin) (all[0].id == truth ? acceptOk : acceptWrong)++;
    }
    std::printf("\n[enrollment: first %d samples -> template, rest are probes]\n", enrollK);
    std::printf("  gallery %zu people (%d with <= %d samples skipped, %d failed template: their probes not counted),"
                " %zu probes\n",
                gallery.entries().size(), skipped, enrollK, failed, probes.size());
    std::printf("  rank-1 identification: %.2f%%  (chance %.2f%%)\n", 100.0 * rank1 / std::max<size_t>(1, probes.size()),
                100.0 / std::max<size_t>(1, gallery.entries().size()));
    describe("probe vs own template", tGen);
    describe("probe vs other templates", tImp);
    describe("enroll sample vs medoid", medoidCos);
    report("probe-template", tGen, tImp);
    if (cfg.acceptCosMin <= 1.f)
      std::printf("  at accept_cos_min %.3f: correct accept %.2f%%, wrong accept %.2f%%, rejected %.2f%%\n",
                  cfg.acceptCosMin, 100.0 * acceptOk / probes.size(), 100.0 * acceptWrong / probes.size(),
                  100.0 * (probes.size() - acceptOk - acceptWrong) / probes.size());

    // ---- suggestions (measured on this data set only)
    const bool enoughImp = tImp.size() * 1e-3 >= 5;
    const Operating op = atFar(tGen, tImp, enoughImp ? 1e-3 : 1e-2);
    std::printf("\nsuggested config.yaml values (from this data set):\n");
    std::printf("  match.accept_cos_min:     %.3f   (probe-template FAR <= %g, FRR %.2f%%)\n", op.thr,
                enoughImp ? 1e-3 : 1e-2, 100 * op.frr);
    std::printf("  template.dup_cos_max:     %.3f   (a new enrollee accepted as an enrolled id is a duplicate)\n", op.thr);
    std::printf("  template.outlier_cos_min: %.3f   (median probe vs other templates: a sample farther from its\n"
                "                                    own medoid than a typical other person is an outlier)\n",
                pct(tImp, 50));

    // ---- optional outputs
    if (a.has("scores")) {
      std::ofstream f(a.get("scores"));
      f << "kind,label,score\n";
      for (float s : gen) f << "sample,genuine," << s << "\n";
      for (float s : imp) f << "sample,impostor," << s << "\n";
      for (float s : tGen) f << "template,genuine," << s << "\n";
      for (float s : tImp) f << "template,impostor," << s << "\n";
      std::printf("scores written to %s\n", a.get("scores").c_str());
    }
    if (a.has("roi-sheet")) {
      const int cols = 10, tile = 112;
      const int rowsN = (static_cast<int>(byPerson.size()) + cols - 1) / cols;
      cv::Mat sheet(rowsN * tile, cols * tile, CV_8UC3, cv::Scalar::all(0));
      int k = 0;
      for (const auto& [person, idx] : byPerson) {
        cv::Mat roi8;
        rois[idx[0]].convertTo(roi8, CV_8UC3);
        cv::cvtColor(roi8, roi8, cv::COLOR_RGB2BGR);
        cv::Mat dst = sheet(cv::Rect((k % cols) * tile, (k / cols) * tile, tile, tile));
        roi8.copyTo(dst);
        cv::putText(dst, person, {3, 12}, cv::FONT_HERSHEY_SIMPLEX, 0.35, {0, 255, 255}, 1, cv::LINE_AA);
        ++k;
      }
      cv::imwrite(a.get("roi-sheet"), sheet);
      std::printf("ROI sheet (first sample of each person) written to %s\n", a.get("roi-sheet").c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
