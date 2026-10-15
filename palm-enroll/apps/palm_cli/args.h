#pragma once
// Minimal "--key value" / "--flag" / positional argument parser for the tools.
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct Args {
  std::map<std::string, std::string> opt;
  std::vector<std::string> pos;

  // flags: options that take no value
  Args(int argc, char** argv, int first, const std::set<std::string>& flags = {}) {
    for (int i = first; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("--", 0) != 0) {
        pos.push_back(a);
      } else if (flags.count(a.substr(2))) {
        opt[a.substr(2)] = "1";
      } else {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        opt[a.substr(2)] = argv[++i];
      }
    }
  }
  bool has(const std::string& k) const { return opt.count(k) > 0; }
  std::string get(const std::string& k, const std::string& def = {}) const {
    const auto it = opt.find(k);
    return it == opt.end() ? def : it->second;
  }
};
