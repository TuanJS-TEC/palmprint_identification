#pragma once
// Enrollment policy: which of the samples captured for one palm are stored.
#include <vector>

#include "palm/vec.h"

namespace palm {

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

}  // namespace palm
