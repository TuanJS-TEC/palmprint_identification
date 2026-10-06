#pragma once
// Reader for the capture log (log_label.csv): ';' or ',' separated, columns
// filename, timestamp, person_id, handedness (MediaPipe label).
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Row {
  std::string file, ts, person, hand;
};

inline std::string trim(std::string s) {
  const char* ws = " \t\r\n";
  s.erase(0, s.find_first_not_of(ws));
  s.erase(s.find_last_not_of(ws) + 1);
  return s;
}

inline std::vector<std::string> split(const std::string& line, char d) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  for (std::string f; std::getline(ss, f, d);) out.push_back(trim(f));
  return out;
}

inline std::vector<Row> readLabels(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::string line;
  std::getline(f, line);
  if (line.rfind("\xEF\xBB\xBF", 0) == 0) line.erase(0, 3);  // UTF-8 BOM
  const char d = line.find(';') != std::string::npos ? ';' : ',';
  const auto head = split(line, d);
  auto col = [&](const char* name) {
    const auto it = std::find(head.begin(), head.end(), name);
    if (it == head.end()) throw std::runtime_error(path + ": no column '" + name + "'");
    return static_cast<size_t>(it - head.begin());
  };
  const size_t cf = col("filename"), ct = col("timestamp"), cp = col("person_id"), ch = col("handedness");
  std::vector<Row> rows;
  while (std::getline(f, line)) {
    const auto v = split(line, d);
    if (v.size() <= std::max({cf, ct, cp, ch}) || v[cf].empty()) continue;
    rows.push_back({v[cf], v[ct], v[cp], v[ch]});
  }
  return rows;
}
