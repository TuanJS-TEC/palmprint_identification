#pragma once
// Internals shared by gallery.cpp and gallery_sqlite.cpp (not installed).
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>

namespace palm::detail {

constexpr uint32_t kMaxSamples = 1024;  // per entry; larger counts mean a corrupt database

inline bool parseUserSeq(const std::string& id, uint32_t& n) {
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

// Moves seq past the number in a "user_<n>" id; other ids are ignored.
inline void bumpSeq(uint32_t& seq, const std::string& id) {
  uint32_t n = 0;
  if (parseUserSeq(id, n)) seq = std::max(seq, n + 1);  // n <= 999999999: no overflow
}

}  // namespace palm::detail
