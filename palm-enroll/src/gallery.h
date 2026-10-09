#pragma once
// Enrollment samples (outlier check), cosine matching against every stored
// sample of a person, and a small on-disk gallery.
#include <cstdint>
#include <string>
#include <vector>

namespace palm {

using Vec = std::vector<float>;

Vec l2normalize(Vec v);
float dot(const Vec& a, const Vec& b);  // cosine for normalised vectors

struct TemplateParams {
  int minValid = 3;              // samples left after outlier removal
  float outlierCosMin = -1.f;    // drop samples with cos(sample, medoid) below; -1 disables
};

struct TemplateResult {
  bool ok = false;
  std::string reason;            // empty when ok
  Vec tmpl;                      // normalised mean of the kept samples (for the log)
  std::vector<int> kept;         // indices into the input: the samples to store
  int medoid = -1;               // index of the sample most similar to the others
  std::vector<float> cosToMedoid;
};

// feats: L2-normalised embeddings of one palm.
TemplateResult buildTemplate(const std::vector<Vec>& feats, const TemplateParams& p);
// The samples of feats that buildTemplate kept (what a gallery entry stores).
std::vector<Vec> keptSamples(const std::vector<Vec>& feats, const TemplateResult& t);

// One embedding per enrollment sample, not their mean: the samples are taken
// at different palm tilts, and a probe matches the closest one. EE-PRnet
// embeddings of one palm drift apart as the palm tilts (cos median 0.98 within
// 5 degrees, 0.75 at 15-20 degrees; output/original, 2026-10-09), so a mean
// over tilts matches none of them well.
struct GalleryEntry {
  std::string id;
  std::vector<Vec> samples;  // L2-normalised
};

// Highest cosine between the probe and any sample of the entry.
float bestScore(const Vec& probe, const GalleryEntry& e);

struct Match {
  std::string id;
  float score = 0.f;
};

// "user_<1-9 digits>" -> n. False for any other id; never throws.
bool parseUserSeq(const std::string& id, uint32_t& n);

class Gallery {
 public:
  explicit Gallery(std::string modelVersion = {}, int dim = 512) : modelVersion_(std::move(modelVersion)), dim_(dim) {}

  // Missing file -> empty gallery. Throws on a corrupt file or a different
  // model version, and then leaves the gallery unchanged. Files of format v1/v2
  // stored one mean template per person: it is loaded as a single sample.
  void load(const std::string& path);
  // Atomic: written to a temporary file of its own, then renamed over path.
  void save(const std::string& path) const;

  // Throws if the entry has no sample or a sample of the wrong dimension.
  void upsert(GalleryEntry e);
  bool remove(const std::string& id);
  const std::vector<GalleryEntry>& entries() const { return entries_; }
  bool contains(const std::string& id) const;
  // Score of an entry = bestScore(probe, entry), highest first.
  std::vector<Match> search(const Vec& probe, int topK, const std::string& excludeId = {}) const;
  // Several probes (the samples of a new enrollment): score of an entry = the
  // highest cosine over every probe / sample pair (duplicate check).
  std::vector<Match> search(const std::vector<Vec>& probes, int topK, const std::string& excludeId = {}) const;

  // Next automatic id, "user_0001", "user_0002", ...: one above every
  // user_<n> id this gallery has held, so the id of a removed person is
  // never given out again (the counter is stored in the file).
  std::string nextId() const;

 private:
  std::string modelVersion_;
  int dim_;
  std::vector<GalleryEntry> entries_;
  uint32_t nextSeq_ = 1;
};

}  // namespace palm
