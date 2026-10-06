#pragma once
// Tiny assertion helpers (no test framework dependency).
#include <cstdio>

inline int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

#define CHECK_MSG(cond, ...)                                              \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);        \
      std::printf(__VA_ARGS__);                                           \
      std::printf("\n");                                                  \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

inline int finish(const char* name) {
  std::printf("%s: %s (%d failures)\n", name, g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
