#include "palm/template.h"

#include <algorithm>

namespace palm {

TemplateResult buildTemplate(const std::vector<Vec>& feats, const TemplateParams& p) {
  TemplateResult r;
  const int n = static_cast<int>(feats.size());
  if (n == 0) return r;
  // medoid
  std::vector<float> meanCos(n, 0.f);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) meanCos[i] += dot(feats[i], feats[j]) / std::max(1, n - 1);
  const int medoid = static_cast<int>(std::max_element(meanCos.begin(), meanCos.end()) - meanCos.begin());

  for (int i = 0; i < n; ++i) {
    const float c = dot(feats[i], feats[medoid]);
    r.cosToMedoid.push_back(c);
    if (c < p.outlierCosMin) continue;
    r.kept.push_back(i);
  }
  r.ok = static_cast<int>(r.kept.size()) >= p.minValid;
  return r;
}

std::vector<Vec> keptSamples(const std::vector<Vec>& feats, const TemplateResult& t) {
  std::vector<Vec> out;
  for (int i : t.kept) out.push_back(feats.at(i));
  return out;
}

}  // namespace palm
