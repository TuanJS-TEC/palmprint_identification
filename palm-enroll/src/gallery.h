#pragma once
// Enrollment samples (outlier check), cosine matching against every stored
// sample of a person, and a small on-disk gallery (a SQLite database).
#include <cstdint>
#include <stdexcept>
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
  bool ok = false;                 // at least minValid samples kept
  std::vector<int> kept;           // indices into the input: the samples to store
  std::vector<float> cosToMedoid;  // per input sample; medoid = the sample most similar to the others
};

// feats: L2-normalised embeddings of one palm. Drops the outliers.
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

struct Match {
  std::string id;
  float score = 0.f;
};

// What went wrong with the gallery file; callers decide what is safe to do
// next: only Corrupt and ModelMismatch say the file itself is of no use.
class GalleryError : public std::runtime_error {
 public:
  enum class Kind {
    Io,             // cannot open / read / write the file
    Busy,           // locked by another process for longer than the busy timeout
    Corrupt,        // not a gallery database, or its content is inconsistent
    ModelMismatch,  // built with another model version or feature dimension
    Version,        // written by a newer program
  };
  GalleryError(Kind kind, const std::string& what) : std::runtime_error(what), kind_(kind) {}
  Kind kind() const { return kind_; }

 private:
  Kind kind_;
};

class Gallery {
 public:
  explicit Gallery(std::string modelVersion = {}, int dim = 512) : modelVersion_(std::move(modelVersion)), dim_(dim) {}

  // Missing or empty file -> empty gallery (nothing is created). Throws
  // GalleryError on an unreadable or corrupt file or a different model
  // version, and then leaves the gallery unchanged.
  void load(const std::string& path);
  // One SQLite transaction that replaces the whole content of path (created if
  // missing): a reader sees the old or the new gallery, a failure keeps the
  // old one. Refuses to overwrite a file that is not a gallery database.
  void save(const std::string& path) const;

  // Throws if the entry has no sample or a sample of the wrong dimension.
  void upsert(GalleryEntry e);
  bool remove(const std::string& id);
  const std::vector<GalleryEntry>& entries() const { return entries_; }
  bool contains(const std::string& id) const;
  // Score of an entry = the highest cosine over every probe / sample pair,
  // highest first. One probe: identification; the samples of a new
  // enrollment: duplicate check.
  std::vector<Match> search(const std::vector<Vec>& probes, int topK, const std::string& excludeId = {}) const;

  // Next automatic id, "user_0001", "user_0002", ...: one above every
  // user_<n> id this gallery has held, so the id of a removed person is
  // never given out again (the counter is stored in the database).
  std::string nextId() const;

 private:
  std::string modelVersion_;
  int dim_;
  std::vector<GalleryEntry> entries_;
  uint32_t nextSeq_ = 1;
};

}  // namespace palm
