#include <cmath>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "gallery.h"

using namespace palm;

static Vec unit(int dim, int axis, float noise = 0.f, int seed = 0) {
  Vec v(dim, 0.f);
  v[axis] = 1.f;
  for (int i = 0; i < dim; ++i) v[i] += noise * std::sin(1.7f * (i + 1) * (seed + 1));
  return l2normalize(v);
}

int main() {
  const int dim = 8;

  // template: outlier rejected, mean of the rest
  std::vector<Vec> f = {unit(dim, 0, 0.1f, 1), unit(dim, 0, 0.1f, 2), unit(dim, 0, 0.1f, 3), unit(dim, 5)};
  TemplateParams p;
  p.minValid = 3;
  p.outlierCosMin = 0.5f;
  TemplateResult t = buildTemplate(f, p);
  CHECK(t.ok);
  CHECK(t.kept.size() == 3 && t.medoid != 3);
  CHECK(std::fabs(dot(t.tmpl, t.tmpl) - 1.f) < 1e-5);
  CHECK(dot(t.tmpl, unit(dim, 0)) > 0.95f);

  p.minValid = 4;
  t = buildTemplate(f, p);
  CHECK(!t.ok && t.reason == "too_few_valid_samples");
  CHECK(!buildTemplate({}, p).ok);

  // gallery: search order, exclude, save/load round trip, version check
  Gallery g("test_v1", dim);
  g.upsert({"a", {unit(dim, 0)}});
  g.upsert({"b", {unit(dim, 1)}});
  g.upsert({"a", {unit(dim, 2), unit(dim, 3)}});  // replaces
  CHECK(g.entries().size() == 2);
  auto m = g.search(unit(dim, 2, 0.05f), 2);
  CHECK(m.size() == 2 && m[0].id == "a" && m[0].score > m[1].score);
  CHECK(g.search(unit(dim, 2), 5, "a").size() == 1);
  // an entry scores its closest sample, not a mean of its samples
  m = g.search(unit(dim, 3), 1);
  CHECK(m[0].id == "a" && std::fabs(m[0].score - 1.f) < 1e-5);
  CHECK(std::fabs(bestScore(unit(dim, 2), g.entries()[0]) - 1.f) < 1e-5);
  // several probes: the highest cosine over every probe / sample pair
  m = g.search(std::vector<Vec>{unit(dim, 4), unit(dim, 1)}, 2);
  CHECK(m.size() == 2 && m[0].id == "b" && std::fabs(m[0].score - 1.f) < 1e-5 && std::fabs(m[1].score) < 1e-5);
  CHECK(g.search(std::vector<Vec>{unit(dim, 1)}, 5, "b").size() == 1);
  // an entry needs samples of the gallery's dimension
  for (const GalleryEntry& bad : {GalleryEntry{"c", {}}, GalleryEntry{"c", {unit(dim, 0), Vec(dim + 1, 0.f)}}}) {
    bool threwBad = false;
    try {
      g.upsert(bad);
    } catch (const std::invalid_argument&) {
      threwBad = true;
    }
    CHECK(threwBad && !g.contains("c"));
  }

  const auto path = (std::filesystem::temp_directory_path() / "palm_test_gallery.bin").string();
  g.save(path);
  Gallery h("test_v1", dim);
  h.load(path);
  CHECK(h.entries().size() == 2 && h.entries()[0].id == "a" && h.entries()[0].samples.size() == 2);
  CHECK(h.entries()[0].samples == g.entries()[0].samples && h.entries()[1].samples == g.entries()[1].samples);
  bool threw = false;
  try {
    Gallery("other_model", dim).load(path);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(h.remove("a") && !h.remove("a") && h.entries().size() == 1);
  std::filesystem::remove(path);

  Gallery empty("test_v1", dim);
  empty.load(path);  // missing file -> empty
  CHECK(empty.entries().empty());

  // id parsing: anything but user_<1-9 digits> is ignored, never throws
  uint32_t n = 0;
  CHECK(parseUserSeq("user_0042", n) && n == 42);
  CHECK(parseUserSeq("user_999999999", n) && n == 999999999u);
  for (const char* bad : {"admin", "user_", "user_12abc", "user_-1", "user_+1", "user_99999999999", "User_1", "user_1 "})
    CHECK_MSG(!parseUserSeq(bad, n), "%s", bad);

  // automatic ids: increasing, a removed id is not given out again, counter saved
  Gallery s("test_v1", dim);
  CHECK(s.nextId() == "user_0001");
  s.upsert({s.nextId(), {unit(dim, 0)}});
  s.upsert({s.nextId(), {unit(dim, 1)}});
  CHECK(s.contains("user_0002") && s.nextId() == "user_0003");
  CHECK(s.remove("user_0002") && s.nextId() == "user_0003");
  s.upsert({"admin", {unit(dim, 2)}});
  s.upsert({"user_12abc", {unit(dim, 3)}});
  CHECK(s.nextId() == "user_0003");
  s.upsert({"user_7", {unit(dim, 4)}});  // typed by hand: counter follows
  CHECK(s.nextId() == "user_0008");
  s.remove("user_7");
  s.save(path);
  Gallery s2("test_v1", dim);
  s2.load(path);
  CHECK(s2.entries().size() == 3 && s2.nextId() == "user_0008");
  for (const auto& e : std::filesystem::directory_iterator(std::filesystem::path(path).parent_path()))
    CHECK_MSG(e.path().filename().string().rfind("palm_test_gallery.bin.tmp", 0) != 0, "left over %s",
              e.path().string().c_str());

  // version 1 files (no counter) still load; the counter comes from the ids
  {
    std::ofstream v1(path, std::ios::binary | std::ios::trunc);
    const auto u32 = [&](uint32_t v) { v1.write(reinterpret_cast<const char*>(&v), 4); };
    const auto str = [&](const std::string& t) {
      u32(static_cast<uint32_t>(t.size()));
      v1.write(t.data(), static_cast<std::streamsize>(t.size()));
    };
    v1.write("PALMGAL1", 8);
    str("test_v1");
    u32(dim);
    u32(2);
    for (const char* who : {"user_05", "bob"}) {
      str(who);
      u32(5);
      const Vec t = unit(dim, 1);
      v1.write(reinterpret_cast<const char*>(t.data()), dim * sizeof(float));
    }
  }
  Gallery old("test_v1", dim);
  old.load(path);
  CHECK(old.entries().size() == 2 && old.entries()[1].id == "bob" && old.nextId() == "user_0006");
  CHECK(old.entries()[1].samples.size() == 1 && old.entries()[1].samples[0] == unit(dim, 1));  // the mean template

  // a v3 entry with a sample count of 0 is corrupt
  {
    std::ofstream v3(path, std::ios::binary | std::ios::trunc);
    const auto u32 = [&](uint32_t v) { v3.write(reinterpret_cast<const char*>(&v), 4); };
    v3.write("PALMGAL3", 8);
    u32(7);
    v3.write("test_v1", 7);
    u32(dim);
    u32(1);
    u32(1);
    u32(1);
    v3.write("x", 1);
    u32(0);
  }
  threw = false;
  try {
    old.load(path);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw && old.entries().size() == 2);

  // a corrupt file throws and leaves the gallery as it was
  { std::ofstream(path, std::ios::binary | std::ios::trunc) << "PALMGAL2 truncated"; }
  threw = false;
  try {
    old.load(path);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw && old.entries().size() == 2 && old.nextId() == "user_0006");
  std::filesystem::remove(path);

  return finish("test_gallery");
}
