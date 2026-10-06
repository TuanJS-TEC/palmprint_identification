#pragma once
// Enrollment template (several samples -> one L2-normalised 512-d vector),
// cosine matching and a small on-disk gallery.
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
  Vec tmpl;                      // normalised mean of the kept samples
  std::vector<int> kept;         // indices into the input
  int medoid = -1;               // index of the sample most similar to the others
  std::vector<float> cosToMedoid;
};

// feats: L2-normalised embeddings of one palm.
TemplateResult buildTemplate(const std::vector<Vec>& feats, const TemplateParams& p);

struct GalleryEntry {
  std::string id;
  int numSamples = 0;
  Vec tmpl;
};

struct Match {
  std::string id;
  float score = 0.f;
};

class Gallery {
 public:
  explicit Gallery(std::string modelVersion = {}, int dim = 512) : modelVersion_(std::move(modelVersion)), dim_(dim) {}

  // Missing file -> empty gallery. Throws on a corrupt file or a different model version.
  void load(const std::string& path);
  void save(const std::string& path) const;  // atomic (write + rename)

  void upsert(GalleryEntry e);
  bool remove(const std::string& id);
  const std::vector<GalleryEntry>& entries() const { return entries_; }
  std::vector<Match> search(const Vec& probe, int topK, const std::string& excludeId = {}) const;

 private:
  std::string modelVersion_;
  int dim_;
  std::vector<GalleryEntry> entries_;
};

}  // namespace palm
