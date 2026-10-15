#include "palm/vec.h"

#include <cmath>
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

}  // namespace palm
