#include "palm/gallery.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "gallery_internal.h"

namespace palm {

using detail::bumpSeq;
using detail::kMaxSamples;

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

std::vector<Match> Gallery::search(const std::vector<Vec>& probes, int topK, const std::string& excludeId) const {
  std::vector<Match> m;
  for (const auto& e : entries_) {
    if (e.id == excludeId) continue;
    float best = -2.f;  // below any cosine
    for (const Vec& p : probes)
      for (const Vec& v : e.samples) best = std::max(best, dot(p, v));
    m.push_back({e.id, best});
  }
  std::sort(m.begin(), m.end(), [](const Match& a, const Match& b) { return a.score > b.score; });
  if (static_cast<int>(m.size()) > topK) m.resize(std::max(0, topK));
  return m;
}

}  // namespace palm
