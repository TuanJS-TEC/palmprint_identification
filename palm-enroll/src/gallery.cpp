#include "gallery.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

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

// ---------------------------------------------------------------- storage
namespace {
constexpr char kMagic[8] = {'P', 'A', 'L', 'M', 'G', 'A', 'L', '1'};

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
  entries_.clear();
  std::ifstream f(path, std::ios::binary);
  if (!f) return;
  char magic[8];
  f.read(magic, sizeof magic);
  if (!f || !std::equal(magic, magic + 8, kMagic)) throw std::runtime_error("gallery: not a palm gallery file: " + path);
  const std::string version = getStr(f);
  const int dim = static_cast<int>(getU32(f));
  const uint32_t count = getU32(f);
  if (version != modelVersion_ || dim != dim_)
    throw std::runtime_error("gallery: built with model '" + version + "' (dim " + std::to_string(dim) +
                             "), current model is '" + modelVersion_ + "'; re-enroll");
  for (uint32_t k = 0; k < count; ++k) {
    GalleryEntry e;
    e.id = getStr(f);
    e.numSamples = static_cast<int>(getU32(f));
    e.tmpl.resize(dim_);
    f.read(reinterpret_cast<char*>(e.tmpl.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
    if (!f) throw std::runtime_error("gallery: truncated file: " + path);
    entries_.push_back(std::move(e));
  }
}

void Gallery::save(const std::string& path) const {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f.write(kMagic, sizeof kMagic);
    putStr(f, modelVersion_);
    putU32(f, static_cast<uint32_t>(dim_));
    putU32(f, static_cast<uint32_t>(entries_.size()));
    for (const auto& e : entries_) {
      putStr(f, e.id);
      putU32(f, static_cast<uint32_t>(e.numSamples));
      f.write(reinterpret_cast<const char*>(e.tmpl.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
    }
    if (!f) throw std::runtime_error("gallery: cannot write " + tmp);
  }
  std::filesystem::rename(tmp, path);
}

void Gallery::upsert(GalleryEntry e) {
  if (static_cast<int>(e.tmpl.size()) != dim_) throw std::invalid_argument("gallery: template dimension mismatch");
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

std::vector<Match> Gallery::search(const Vec& probe, int topK, const std::string& excludeId) const {
  std::vector<Match> m;
  for (const auto& e : entries_)
    if (e.id != excludeId) m.push_back({e.id, dot(probe, e.tmpl)});
  std::sort(m.begin(), m.end(), [](const Match& a, const Match& b) { return a.score > b.score; });
  if (static_cast<int>(m.size()) > topK) m.resize(topK);
  return m;
}

}  // namespace palm
