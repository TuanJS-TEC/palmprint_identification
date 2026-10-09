#include "gallery.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>

namespace palm {

Vec l2normalize(Vec v) {
  const float n = std::sqrt(dot(v, v));
  if (n > 0.f)
    for (float& x : v) x /= n;
  return v;
}

float dot(const Vec& a, const Vec& b) {
  if (a.size() != b.size()) throw std::invalid_argument("dot: size mismatch");
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.f);
}

TemplateResult buildTemplate(const std::vector<Vec>& feats, const TemplateParams& p) {
  TemplateResult r;
  const int n = static_cast<int>(feats.size());
  if (n == 0) {
    r.reason = "no_samples";
    return r;
  }
  // medoid: the sample with the highest mean cosine to the others
  std::vector<float> meanCos(n, 0.f);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) meanCos[i] += dot(feats[i], feats[j]) / std::max(1, n - 1);
  r.medoid = static_cast<int>(std::max_element(meanCos.begin(), meanCos.end()) - meanCos.begin());

  Vec sum(feats[0].size(), 0.f);
  for (int i = 0; i < n; ++i) {
    const float c = dot(feats[i], feats[r.medoid]);
    r.cosToMedoid.push_back(c);
    if (c < p.outlierCosMin) continue;
    r.kept.push_back(i);
    for (size_t k = 0; k < sum.size(); ++k) sum[k] += feats[i][k];
  }
  if (static_cast<int>(r.kept.size()) < p.minValid) {
    r.reason = "too_few_valid_samples";
    return r;
  }
  r.tmpl = l2normalize(std::move(sum));
  r.ok = true;
  return r;
}

std::vector<Vec> keptSamples(const std::vector<Vec>& feats, const TemplateResult& t) {
  std::vector<Vec> out;
  for (int i : t.kept) out.push_back(feats.at(i));
  return out;
}

bool parseUserSeq(const std::string& id, uint32_t& n) {
  constexpr std::string_view kPrefix = "user_";
  if (id.size() <= kPrefix.size() || id.size() > kPrefix.size() + 9 || id.compare(0, kPrefix.size(), kPrefix) != 0)
    return false;
  const char* first = id.data() + kPrefix.size();
  const char* last = id.data() + id.size();
  if (!std::all_of(first, last, [](char c) { return c >= '0' && c <= '9'; })) return false;
  uint32_t v = 0;
  const auto r = std::from_chars(first, last, v);
  if (r.ec != std::errc() || r.ptr != last) return false;
  n = v;
  return true;
}

// ---------------------------------------------------------------- storage
namespace {
// v1: magic, model version, dim, count, entries (id, sample count, mean template)
// v2: magic, model version, dim, next id sequence, count, entries as v1
// v3: magic, model version, dim, next id sequence, count, entries (id, n, n samples)
constexpr char kMagicV1[8] = {'P', 'A', 'L', 'M', 'G', 'A', 'L', '1'};
constexpr char kMagicV2[8] = {'P', 'A', 'L', 'M', 'G', 'A', 'L', '2'};
constexpr char kMagicV3[8] = {'P', 'A', 'L', 'M', 'G', 'A', 'L', '3'};
constexpr uint32_t kMaxSamples = 1024;  // per entry; larger counts mean a corrupt file

void bumpSeq(uint32_t& seq, const std::string& id) {
  uint32_t n = 0;
  if (parseUserSeq(id, n)) seq = std::max(seq, n + 1);  // n <= 999999999: no overflow
}

void putU32(std::ostream& o, uint32_t v) { o.write(reinterpret_cast<const char*>(&v), sizeof v); }
void putStr(std::ostream& o, const std::string& s) {
  putU32(o, static_cast<uint32_t>(s.size()));
  o.write(s.data(), static_cast<std::streamsize>(s.size()));
}
uint32_t getU32(std::istream& i) {
  uint32_t v = 0;
  i.read(reinterpret_cast<char*>(&v), sizeof v);
  return v;
}
std::string getStr(std::istream& i) {
  const uint32_t n = getU32(i);
  if (!i || n > (1u << 20)) throw std::runtime_error("gallery: corrupt string length");
  std::string s(n, '\0');
  i.read(s.data(), static_cast<std::streamsize>(n));
  return s;
}
}  // namespace

void Gallery::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    entries_.clear();
    nextSeq_ = 1;
    return;
  }
  // parsed into locals: a corrupt file leaves the gallery as it was
  char magic[8];
  f.read(magic, sizeof magic);
  const bool v1 = f && std::equal(magic, magic + 8, kMagicV1), v2 = f && std::equal(magic, magic + 8, kMagicV2),
             v3 = f && std::equal(magic, magic + 8, kMagicV3);
  if (!v1 && !v2 && !v3) throw std::runtime_error("gallery: not a palm gallery file: " + path);
  const std::string version = getStr(f);
  const int dim = static_cast<int>(getU32(f));
  uint32_t seq = v1 ? 1 : getU32(f);
  const uint32_t count = getU32(f);
  if (!f) throw std::runtime_error("gallery: truncated file: " + path);
  if (version != modelVersion_ || dim != dim_)
    throw std::runtime_error("gallery: built with model '" + version + "' (dim " + std::to_string(dim) +
                             "), current model is '" + modelVersion_ + "'; re-enroll");
  std::vector<GalleryEntry> entries;
  for (uint32_t k = 0; k < count; ++k) {
    GalleryEntry e;
    e.id = getStr(f);
    const uint32_t n = getU32(f);
    if (!f || (v3 && (n == 0 || n > kMaxSamples))) throw std::runtime_error("gallery: corrupt sample count: " + path);
    e.samples.resize(v3 ? n : 1);  // v1/v2: n is the enrollment count, one mean template follows
    for (Vec& v : e.samples) {
      v.resize(dim_);
      f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
    }
    if (!f) throw std::runtime_error("gallery: truncated file: " + path);
    bumpSeq(seq, e.id);
    entries.push_back(std::move(e));
  }
  entries_ = std::move(entries);
  nextSeq_ = std::max<uint32_t>(seq, 1);
}

void Gallery::save(const std::string& path) const {
  // A temporary name of its own: two processes saving at once must not write
  // into the same file. rename() then replaces path atomically, so a reader
  // sees either the old or the new gallery, never a partial one.
  std::random_device rd;
  char suffix[24];
  std::snprintf(suffix, sizeof suffix, ".tmp.%08x%08x", rd(), rd());
  const std::string tmp = path + suffix;
  try {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f.write(kMagicV3, sizeof kMagicV3);
    putStr(f, modelVersion_);
    putU32(f, static_cast<uint32_t>(dim_));
    putU32(f, nextSeq_);
    putU32(f, static_cast<uint32_t>(entries_.size()));
    for (const auto& e : entries_) {
      putStr(f, e.id);
      putU32(f, static_cast<uint32_t>(e.samples.size()));
      for (const Vec& v : e.samples)
        f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
    }
    f.close();  // flush errors show up here, before the rename
    if (!f) throw std::runtime_error("gallery: cannot write " + tmp);
    std::filesystem::rename(tmp, path);
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    throw;
  }
}

bool Gallery::contains(const std::string& id) const {
  return std::any_of(entries_.begin(), entries_.end(), [&](const GalleryEntry& e) { return e.id == id; });
}

std::string Gallery::nextId() const {
  char id[24];
  std::snprintf(id, sizeof id, "user_%04u", static_cast<unsigned>(nextSeq_));
  return id;
}

void Gallery::upsert(GalleryEntry e) {
  if (e.samples.empty() || e.samples.size() > kMaxSamples)
    throw std::invalid_argument("gallery: an entry needs 1.." + std::to_string(kMaxSamples) + " samples");
  for (const Vec& v : e.samples)
    if (static_cast<int>(v.size()) != dim_) throw std::invalid_argument("gallery: sample dimension mismatch");
  bumpSeq(nextSeq_, e.id);
  for (auto& x : entries_)
    if (x.id == e.id) {
      x = std::move(e);
      return;
    }
  entries_.push_back(std::move(e));
}

bool Gallery::remove(const std::string& id) {
  const auto it = std::remove_if(entries_.begin(), entries_.end(), [&](const GalleryEntry& e) { return e.id == id; });
  const bool found = it != entries_.end();
  entries_.erase(it, entries_.end());
  return found;
}

float bestScore(const Vec& probe, const GalleryEntry& e) {
  float best = -2.f;  // below any cosine
  for (const Vec& v : e.samples) best = std::max(best, dot(probe, v));
  return best;
}

namespace {
void sortTop(std::vector<Match>& m, int topK) {
  std::sort(m.begin(), m.end(), [](const Match& a, const Match& b) { return a.score > b.score; });
  if (static_cast<int>(m.size()) > topK) m.resize(std::max(0, topK));
}
}  // namespace

std::vector<Match> Gallery::search(const Vec& probe, int topK, const std::string& excludeId) const {
  std::vector<Match> m;
  for (const auto& e : entries_)
    if (e.id != excludeId) m.push_back({e.id, bestScore(probe, e)});
  sortTop(m, topK);
  return m;
}

std::vector<Match> Gallery::search(const std::vector<Vec>& probes, int topK, const std::string& excludeId) const {
  std::vector<Match> m;
  for (const auto& e : entries_) {
    if (e.id == excludeId) continue;
    float best = -2.f;
    for (const Vec& p : probes) best = std::max(best, bestScore(p, e));
    m.push_back({e.id, best});
  }
  sortTop(m, topK);
  return m;
}

}  // namespace palm
