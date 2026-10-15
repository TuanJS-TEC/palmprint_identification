#pragma once
// Embedding vectors: the plain float vector type and its two operations.
#include <vector>

namespace palm {

using Vec = std::vector<float>;

Vec l2normalize(Vec v);
float dot(const Vec& a, const Vec& b);  // cosine for normalised vectors

}  // namespace palm
