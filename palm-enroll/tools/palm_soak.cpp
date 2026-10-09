// palm_soak: long-run memory and CPU test of the live path, the loop of
// palm_gui's CameraWorker in identify mode without the Qt preview: every frame
// goes through the hand detector and the live checks; frames that pass are
// embedded (EE-PRnet), L2-normalised and matched against the gallery.
//
// The camera is simulated: a set of frames (every --frames-every-th file of
// DIR with the camera size of config.yaml, plus one empty grey frame) is
// decoded once, and each iteration copies one into the frame buffer as
// Grabber::next does. A frame is held for --hold iterations, so the motion
// check sees a moving hand on the first and a still one on the others, as
// with a hand held in front of the camera.
//
// Memory is read from /proc/self/status (RSS, peak RSS), mallinfo2 (heap in
// use) and the malloc wrappers below, which count every allocation of every
// thread (OpenCV, ONNX Runtime, the C++ library). Samples go to --csv every
// --sample-every frames.
#include <malloc.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
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

// ---------------------------------------------------------------- malloc wrappers
// ELF symbol interposition: these replace malloc & co. for the whole process
// and forward to glibc's own implementation. Blocks are counted by their
// usable size, so a block weighs the same on malloc and on free.
extern "C" {
void* __libc_malloc(size_t);
void __libc_free(void*);
void* __libc_calloc(size_t, size_t);
void* __libc_realloc(void*, size_t);
void* __libc_memalign(size_t, size_t);
void* __libc_valloc(size_t);
void* __libc_pvalloc(size_t);
}

namespace heap {

std::atomic<long long> live{0};   // bytes allocated and not freed
std::atomic<long long> peak{0};   // highest `live` since the last reset
std::atomic<long long> bytes{0};  // bytes ever allocated
std::atomic<long long> count{0};  // allocations ever made

long long sizeOf(void* p) { return p ? static_cast<long long>(malloc_usable_size(p)) : 0; }

void added(void* p) {
  if (!p) return;
  const long long n = sizeOf(p);
  const long long v = live.fetch_add(n, std::memory_order_relaxed) + n;
  bytes.fetch_add(n, std::memory_order_relaxed);
  count.fetch_add(1, std::memory_order_relaxed);
  long long pk = peak.load(std::memory_order_relaxed);
  while (v > pk && !peak.compare_exchange_weak(pk, v, std::memory_order_relaxed)) {
  }
}

void removed(long long n) { live.fetch_sub(n, std::memory_order_relaxed); }

}  // namespace heap

extern "C" {
void* malloc(size_t n) noexcept {
  void* p = __libc_malloc(n);
  heap::added(p);
  return p;
}
void free(void* p) noexcept {
  heap::removed(heap::sizeOf(p));
  __libc_free(p);
}
void* calloc(size_t n, size_t size) noexcept {
  void* p = __libc_calloc(n, size);
  heap::added(p);
  return p;
}
void* realloc(void* p, size_t n) noexcept {
  const long long old = heap::sizeOf(p);
  void* q = __libc_realloc(p, n);
  if (q || n == 0) {  // p was released or moved (realloc(p, 0) frees p)
    heap::removed(old);
    heap::added(q);
  }
  return q;
}
void* memalign(size_t align, size_t n) noexcept {
  void* p = __libc_memalign(align, n);
  heap::added(p);
  return p;
}
void* aligned_alloc(size_t align, size_t n) noexcept { return memalign(align, n); }
int posix_memalign(void** out, size_t align, size_t n) noexcept {
  if (align % sizeof(void*) != 0 || (align & (align - 1)) != 0) return EINVAL;
  void* p = __libc_memalign(align, n);
  if (!p) return ENOMEM;
  heap::added(p);
  *out = p;
  return 0;
}
void* valloc(size_t n) noexcept {
  void* p = __libc_valloc(n);
  heap::added(p);
  return p;
}
void* pvalloc(size_t n) noexcept {
  void* p = __libc_pvalloc(n);
  heap::added(p);
  return p;
}
}

// ---------------------------------------------------------------- tool
namespace fs = std::filesystem;
using namespace palm;

namespace {

const char* kUsage = R"(usage: palm_soak [--config config.yaml] --frames DIR [--frames-mirrored] [--frames-every K]
                 [--n 100000] [--warmup 200] [--hold 4] [--steady-from 10000] [--sample-every 500]
                 [--gallery-size 1000] [--det-threads 2] [--csv soak.csv]
  --frames DIR         camera frames (*.jpg, *.png); only those of the camera size in config.yaml
  --frames-mirrored    frames are mirror images of the camera view (as palm_eval)
  --frames-every K     use every K-th file of DIR (default 25)
  --n N                frames measured after the warm-up (default 100000)
  --hold H             iterations each frame is shown for (default 4)
  --steady-from F      first measured frame of the steady state (default 10000)
  --gallery-size G     synthetic gallery searched by accepted frames (default 1000)
  --det-threads T      hand detector threads (default 2, as palm_gui)
)";

using Clock = std::chrono::steady_clock;
constexpr double kMb = 1024.0 * 1024.0;

double msBetween(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

double cpuMs() {  // all threads of the process
  timespec ts{};
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) / 1e6;
}

struct Mem {
  double rss = 0, anon = 0, file = 0, hwm = 0;  // MB, /proc/self/status
  double live = 0;                              // MB, malloc wrappers
  double inuse = 0, arena = 0;                  // MB, mallinfo2: in use / taken from the OS
  int threads = 0;
};

Mem readMem() {
  Mem m;
  std::ifstream f("/proc/self/status");
  std::string line, key;
  while (std::getline(f, line)) {
    std::istringstream in(line);
    double v = 0;
    if (!(in >> key >> v)) continue;  // "Name:", "State:" ... are not numbers
    if (key == "VmRSS:") m.rss = v / 1024;
    else if (key == "RssAnon:") m.anon = v / 1024;
    else if (key == "RssFile:") m.file = v / 1024;
    else if (key == "VmHWM:") m.hwm = v / 1024;
    else if (key == "Threads:") m.threads = static_cast<int>(v);
  }
  m.live = static_cast<double>(heap::live.load()) / kMb;
  const struct mallinfo2 mi = mallinfo2();
  m.inuse = static_cast<double>(mi.uordblks + mi.hblkhd) / kMb;
  m.arena = static_cast<double>(mi.arena + mi.hblkhd) / kMb;
  return m;
}

void resetPeakRss() {  // VmHWM := VmRSS (Linux >= 4.0)
  std::ofstream("/proc/self/clear_refs") << "5";
}

// CPU seconds of each thread (utime + stime), by tid
std::map<int, double> threadCpu() {
  std::map<int, double> r;
  const double tick = static_cast<double>(sysconf(_SC_CLK_TCK));
  for (const auto& e : fs::directory_iterator("/proc/self/task")) {
    std::ifstream f(e.path() / "stat");
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t p = s.rfind(')');  // comm may contain spaces
    if (p == std::string::npos) continue;
    std::istringstream in(s.substr(p + 2));
    std::string field;
    std::vector<std::string> v;
    while (in >> field && v.size() < 13) v.push_back(field);
    if (v.size() < 13) continue;  // fields 3.. : state ... utime (14) stime (15)
    r[std::stoi(e.path().filename().string())] = (std::stod(v[11]) + std::stod(v[12])) / tick;
  }
  return r;
}

std::set<int> threadIds() {
  std::set<int> s;
  for (const auto& [tid, cpu] : threadCpu()) s.insert(tid);
  return s;
}

struct Stats {
  std::vector<double> v;
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

void row(const char* name, const Stats& s, double scale = 1.0) {
  std::printf("  %-34s %7zu  %9.3f  %9.3f  %9.3f  %9.3f  %9.3f\n", name, s.v.size(), s.mean() * scale, s.pct(50) * scale,
              s.pct(95) * scale, s.pct(99) * scale, s.max() * scale);
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

struct Sample {
  long frame = 0;
  double t = 0, cpu = 0;  // s since the end of the warm-up
  Mem mem;
  long long allocs = 0, allocBytes = 0;
  long embedded = 0;
};

// least-squares slope of y over x
double slope(const std::vector<double>& x, const std::vector<double>& y) {
  const double n = static_cast<double>(x.size());
  if (n < 2) return 0.0;
  const double mx = std::accumulate(x.begin(), x.end(), 0.0) / n, my = std::accumulate(y.begin(), y.end(), 0.0) / n;
  double sxy = 0, sxx = 0;
  for (size_t i = 0; i < x.size(); ++i) sxy += (x[i] - mx) * (y[i] - my), sxx += (x[i] - mx) * (x[i] - mx);
  return sxx > 0 ? sxy / sxx : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a(argc, argv, 1, {"frames-mirrored"});
    if (!a.has("frames")) {
      std::cerr << kUsage;
      return 1;
    }
    std::vector<std::pair<std::string, Mem>> phases;
    std::vector<std::pair<std::string, std::set<int>>> threadsAt;  // tids alive after each phase
    auto phase = [&](const std::string& name) {
      phases.emplace_back(name, readMem());
      threadsAt.emplace_back(name, threadIds());
    };
    phase("start (libraries loaded)");

    const Config cfg = loadConfig(a.get("config", "config.yaml"));
    const long n = std::stol(a.get("n", "100000"));
    const long warmup = std::stol(a.get("warmup", "200"));
    const long hold = std::max(1L, std::stol(a.get("hold", "4")));
    const long steadyFrom = std::min(n - 1, std::stol(a.get("steady-from", "10000")));
    const long sampleEvery = std::max(1L, std::stol(a.get("sample-every", "500")));
    const size_t every = std::max(1UL, std::stoul(a.get("frames-every", "25")));
    const int galleryN = std::stoi(a.get("gallery-size", "1000"));
    const int detThreads = std::stoi(a.get("det-threads", "2"));
    const cv::Size camSize(cfg.camera.width, cfg.camera.height);

    // ---- simulated camera: decoded frames, kept in memory
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(a.get("frames"))) {
      const auto ext = e.path().extension();
      if (e.is_regular_file() && (ext == ".jpg" || ext == ".png")) files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    std::vector<cv::Mat> frames;
    for (size_t i = 0; i < files.size(); i += every) {
      cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
      if (img.size() != camSize) continue;
      if (a.has("frames-mirrored")) cv::flip(img, img, 1);
      frames.push_back(img);
    }
    if (frames.empty()) throw std::runtime_error("no frame of the camera size in " + a.get("frames"));
    frames.emplace_back(camSize, CV_8UC3, cv::Scalar::all(128));  // nobody in front of the camera

    // per-frame records, filled now so their pages do not show up as growth later
    std::vector<double> wall(n, -1), cpu(n, -1), msDet(n, -1), msEmb(n, -1), cpuDet(n, -1), cpuEmb(n, -1);
    std::vector<double> transient(n, -1), allocB(n, -1), allocN(n, -1), net(n, -1);
    std::vector<char> embedded(n, 0);
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(n / sampleEvery + 2));
    phase("frames decoded (test harness)");

    // ---- the algorithm, as CameraWorker::run builds it
    auto det = std::make_unique<HandDetector>(cfg.hand.palmOnnx, cfg.hand.landmarkOnnx, detThreads);
    phase("hand detector loaded");
    const double hwmBeforeModel = readMem().hwm;
    auto model = std::make_unique<PalmModel>(cfg.model);
    phase("EE-PRnet loaded");
    const double hwmModelLoad = readMem().hwm;
    const Gallery gallery = syntheticGallery(cfg.modelVersion, model->featureDim(), galleryN);
    phase("gallery built");

    cv::Mat frame;  // the frame buffer Grabber::next copies into
    HandLandmarks prev;
    std::map<HandState, long> states;
    double sink = 0;
    long nEmbedded = 0;
    std::map<int, double> cpuAtStart;
    Clock::time_point tStart;
    double cpuStart = 0;

    for (long i = -warmup; i < n; ++i) {
      if (i == 0) {
        phase("after warm-up (" + std::to_string(warmup) + " frames)");
        resetPeakRss();
        cpuAtStart = threadCpu();
        tStart = Clock::now();
        cpuStart = cpuMs();
      }
      const cv::Mat& src = frames[static_cast<size_t>(((i + warmup) / hold) % static_cast<long>(frames.size()))];
      const long long live0 = heap::live.load(), bytes0 = heap::bytes.load(), count0 = heap::count.load();
      heap::peak.store(live0);
      const double c0 = cpuMs();
      const auto t0 = Clock::now();

      src.copyTo(frame);
      const HandLandmarks h = det->detect(frame);
      const HandCheck chk = checkHand(frame, h, prev, cfg.capture);
      prev = h;
      const double c1 = cpuMs();
      const auto t1 = Clock::now();
      const bool ok = chk.state == HandState::Ok;
      if (ok) {
        const Embedding e = model->run(frame(chk.box), false);  // camera frames are not mirrored
        const Vec feat = l2normalize(e.feature);
        const auto top = gallery.search(feat, 2);  // as MainWindow in identify mode
        sink += top.empty() ? 0.0 : top[0].score;
      }
      const double c2 = cpuMs();
      const auto t2 = Clock::now();
      if (i < 0) continue;

      const auto k = static_cast<size_t>(i);
      wall[k] = msBetween(t0, t2);
      cpu[k] = c2 - c0;
      msDet[k] = msBetween(t0, t1);
      cpuDet[k] = c1 - c0;
      msEmb[k] = msBetween(t1, t2);
      cpuEmb[k] = c2 - c1;
      transient[k] = static_cast<double>(heap::peak.load() - live0);
      allocB[k] = static_cast<double>(heap::bytes.load() - bytes0);
      allocN[k] = static_cast<double>(heap::count.load() - count0);
      net[k] = static_cast<double>(heap::live.load() - live0);
      embedded[k] = ok;
      nEmbedded += ok;
      ++states[chk.state];

      if (i % sampleEvery == 0 || i == n - 1) {
        Sample s;
        s.frame = i;
        s.t = msBetween(tStart, Clock::now()) / 1e3;
        s.cpu = (cpuMs() - cpuStart) / 1e3;
        s.mem = readMem();
        s.allocs = heap::count.load();
        s.allocBytes = heap::bytes.load();
        s.embedded = nEmbedded;
        samples.push_back(s);
      }
    }
    const double wallS = msBetween(tStart, Clock::now()) / 1e3, cpuS = (cpuMs() - cpuStart) / 1e3;
    const std::map<int, double> cpuAtEnd = threadCpu();
    phase("end of loop");
    const double hwmLoop = phases.back().second.hwm;
    model.reset();
    det.reset();
    phase("models released");

    // ---- report
    std::printf("palm_soak  model %s  threads %d  detector threads %d\n", cfg.model.onnxPath.c_str(), cfg.model.threads,
                detThreads);
    std::printf("frames: %zu decoded %dx%d (every %zu-th of %zu files) + 1 empty, each held %ld iterations\n",
                frames.size() - 1, camSize.width, camSize.height, every, files.size(), hold);
    std::printf("measured %ld frames after %ld warm-up frames in %.1f s (%.1f frames/s); embedded %ld (%.1f%%)\n", n,
                warmup, wallS, static_cast<double>(n) / wallS, nEmbedded, 100.0 * static_cast<double>(nEmbedded) / static_cast<double>(n));
    for (const auto& [s, c] : states) std::printf("  state %-44s %ld\n", stateMessage(s), c);

    std::printf("\nmemory by phase (MB)\n  %-36s %8s %8s %8s %8s %10s %10s %10s %7s\n", "phase", "RSS", "anon", "file",
                "peakRSS", "heap-live", "heap-inuse", "heap-arena", "threads");
    for (const auto& [name, m] : phases)
      std::printf("  %-36s %8.1f %8.1f %8.1f %8.1f %10.1f %10.1f %10.1f %7d\n", name.c_str(), m.rss, m.anon, m.file, m.hwm,
                  m.live, m.inuse, m.arena, m.threads);
    std::printf("  peak RSS before EE-PRnet load %.1f, after it %.1f; peak RSS during the measured loop %.1f\n",
                hwmBeforeModel, hwmModelLoad, hwmLoop);

    std::printf("\nthreads created by phase\n");
    std::set<int> seen;
    std::map<int, std::string> origin;
    for (const auto& [name, tids] : threadsAt) {
      std::vector<int> added;
      for (int t : tids)
        if (seen.insert(t).second) added.push_back(t), origin[t] = name;
      if (!added.empty()) std::printf("  %-36s %zu\n", name.c_str(), added.size());
    }

    std::printf("\nCPU over the measured loop: %.1f CPU-s in %.1f s wall = %.2f cores on average (%ld logical CPUs online)\n",
                cpuS, wallS, cpuS / wallS, sysconf(_SC_NPROCESSORS_ONLN));
    std::map<std::string, std::pair<int, double>> byOrigin;  // threads busy > 1% of the loop, CPU-s
    for (const auto& [tid, c] : cpuAtEnd) {
      const auto it = cpuAtStart.find(tid);
      const double d = c - (it == cpuAtStart.end() ? 0.0 : it->second);
      auto& b = byOrigin[origin.count(tid) ? origin[tid] : "during the loop"];
      if (d > 0.01 * wallS) ++b.first;
      b.second += d;
    }
    for (const auto& [name, b] : byOrigin)
      std::printf("  threads created at %-36s busy threads %2d  %8.1f CPU-s  %5.2f cores\n", name.c_str(), b.first, b.second,
                  b.second / wallS);
    double peakCores = 0;
    for (size_t s = 1; s < samples.size(); ++s)
      peakCores = std::max(peakCores, (samples[s].cpu - samples[s - 1].cpu) / (samples[s].t - samples[s - 1].t));
    std::printf("  highest %ld-frame interval: %.2f cores\n", sampleEvery, peakCores);

    Stats sWall, sWallE, sWallN, sDet, sEmb, sCoreDet, sCoreEmb;
    Stats tE, tN, bE, bN, cE, cN;
    double sumNet = 0, sumDetMs = 0, sumDetCpu = 0, sumEmbMs = 0, sumEmbCpu = 0;
    for (size_t k = 0; k < static_cast<size_t>(n); ++k) {
      sWall.v.push_back(wall[k]);
      sDet.v.push_back(msDet[k]);
      sumDetMs += msDet[k], sumDetCpu += cpuDet[k];
      sumNet += net[k];
      if (embedded[k]) {
        sWallE.v.push_back(wall[k]);
        sEmb.v.push_back(msEmb[k]);
        sumEmbMs += msEmb[k], sumEmbCpu += cpuEmb[k];
        tE.v.push_back(transient[k]), bE.v.push_back(allocB[k]), cE.v.push_back(allocN[k]);
      } else {
        sWallN.v.push_back(wall[k]);
        tN.v.push_back(transient[k]), bN.v.push_back(allocB[k]), cN.v.push_back(allocN[k]);
      }
    }
    std::printf("  CPU per stage: detect+checks %.2f cores, embed+match %.2f cores (CPU time / wall time)\n",
                sumDetCpu / sumDetMs, sumEmbMs > 0 ? sumEmbCpu / sumEmbMs : 0.0);

    std::printf("\nlatency (ms per frame)\n  %-34s %7s  %9s  %9s  %9s  %9s  %9s\n", "", "n", "mean", "p50", "p95", "p99", "max");
    row("frame -> match, all frames", sWall);
    row("frame -> match, embedded frames", sWallE);
    row("frame, not embedded", sWallN);
    row("detect + checks", sDet);
    row("embed + normalise + match", sEmb);

    std::printf("\ndynamic memory per frame (heap, all threads)\n  %-34s %7s  %9s  %9s  %9s  %9s  %9s\n", "", "n", "mean",
                "p50", "p95", "p99", "max");
    row("embedded: transient peak (MB)", tE, 1 / kMb);
    row("embedded: allocated (MB)", bE, 1 / kMb);
    row("embedded: allocations (count)", cE);
    row("not embedded: transient peak (MB)", tN, 1 / kMb);
    row("not embedded: allocated (MB)", bN, 1 / kMb);
    row("not embedded: allocations (count)", cN);
    std::printf("  net heap change over all measured frames: %+.0f bytes\n", sumNet);

    // steady state: samples from steadyFrom on
    std::vector<double> x, yRss, yAnon, yLive;
    const Sample* s0 = nullptr;
    double rssMin = 1e18, rssMax = 0, liveMin = 1e18, liveMax = 0;
    for (const auto& s : samples) {
      if (s.frame < steadyFrom) continue;
      if (!s0) s0 = &s;
      x.push_back(static_cast<double>(s.frame));
      yRss.push_back(s.mem.rss), yAnon.push_back(s.mem.anon), yLive.push_back(s.mem.live);
      rssMin = std::min(rssMin, s.mem.rss), rssMax = std::max(rssMax, s.mem.rss);
      liveMin = std::min(liveMin, s.mem.live), liveMax = std::max(liveMax, s.mem.live);
    }
    const Sample& s1 = samples.back();
    if (s0) {
      std::printf("\nsteady state, frames %ld..%ld (%zu samples)\n", s0->frame, s1.frame, x.size());
      std::printf("  RSS       %8.2f -> %8.2f MB  growth %+7.2f MB  range %.2f..%.2f  slope %+.4f MB / 10k frames\n",
                  s0->mem.rss, s1.mem.rss, s1.mem.rss - s0->mem.rss, rssMin, rssMax, 1e4 * slope(x, yRss));
      std::printf("  RSS anon  %8.2f -> %8.2f MB  growth %+7.2f MB  slope %+.4f MB / 10k frames\n", s0->mem.anon,
                  s1.mem.anon, s1.mem.anon - s0->mem.anon, 1e4 * slope(x, yAnon));
      std::printf("  heap live %8.2f -> %8.2f MB  growth %+7.3f MB  range %.2f..%.2f  slope %+.4f MB / 10k frames\n",
                  s0->mem.live, s1.mem.live, s1.mem.live - s0->mem.live, liveMin, liveMax, 1e4 * slope(x, yLive));
    }

    if (a.has("csv")) {
      std::ofstream f(a.get("csv"));
      f << "frame,t_s,cpu_s,rss_mb,rss_anon_mb,rss_file_mb,peak_rss_mb,heap_live_mb,heap_inuse_mb,heap_arena_mb,threads,"
           "allocs,alloc_gb,embedded\n";
      for (const auto& s : samples)
        f << s.frame << ',' << s.t << ',' << s.cpu << ',' << s.mem.rss << ',' << s.mem.anon << ',' << s.mem.file << ','
          << s.mem.hwm << ',' << s.mem.live << ',' << s.mem.inuse << ',' << s.mem.arena << ',' << s.mem.threads << ','
          << s.allocs << ',' << static_cast<double>(s.allocBytes) / 1e9 << ',' << s.embedded << '\n';
      std::printf("\nsamples: %s\n", a.get("csv").c_str());
    }
    return sink == -1.0;  // keeps the match results alive
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
