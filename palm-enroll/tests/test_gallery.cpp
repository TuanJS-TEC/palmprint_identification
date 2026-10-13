#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "check.h"
#include "gallery.h"
#include "sqlite3.h"

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

  const auto path = (std::filesystem::temp_directory_path() / "palm_test_gallery.db").string();
  std::filesystem::remove(path);
  const auto throwsKind = [](auto&& fn, GalleryError::Kind kind) {
    try {
      fn();
    } catch (const GalleryError& e) {
      return e.kind() == kind;
    } catch (...) {
    }
    return false;
  };

  g.save(path);
  Gallery h("test_v1", dim);
  h.load(path);
  CHECK(h.entries().size() == 2 && h.entries()[0].id == "a" && h.entries()[0].samples.size() == 2);
  CHECK(h.entries()[0].samples == g.entries()[0].samples && h.entries()[1].samples == g.entries()[1].samples);
  CHECK(throwsKind([&] { Gallery("other_model", dim).load(path); }, GalleryError::Kind::ModelMismatch));
  CHECK(throwsKind([&] { Gallery("test_v1", dim + 1).load(path); }, GalleryError::Kind::ModelMismatch));
  CHECK(h.remove("a") && !h.remove("a") && h.entries().size() == 1);
  h.save(path);  // replaces the content, not appends
  Gallery h2("test_v1", dim);
  h2.load(path);
  CHECK(h2.entries().size() == 1 && h2.entries()[0].id == "b");
  std::filesystem::remove(path);

  Gallery empty("test_v1", dim);
  empty.load(path);  // missing file -> empty, and nothing is created
  CHECK(empty.entries().empty() && !std::filesystem::exists(path));
  { std::ofstream(path, std::ios::binary | std::ios::trunc); }  // 0 bytes -> empty as well
  empty.load(path);
  CHECK(empty.entries().empty());
  std::filesystem::remove(path);

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
  CHECK(s2.entries()[0].id == "user_0001" && s2.entries()[1].id == "admin" && s2.entries()[2].id == "user_12abc");
  // SQLite leaves its journal beside the database only while a write is running
  for (const auto& e : std::filesystem::directory_iterator(std::filesystem::path(path).parent_path()))
    CHECK_MSG(e.path().filename().string().rfind("palm_test_gallery.db-", 0) != 0, "left over %s",
              e.path().string().c_str());

  // a database edited behind the Gallery's back: every inconsistency is Corrupt, the gallery stays as it was
  const auto corrupt = [&](const char* what, const char* sql) {
    std::filesystem::remove(path);
    s.save(path);
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(path.c_str(), &db) == SQLITE_OK);
    char* err = nullptr;
    CHECK_MSG(sqlite3_exec(db, sql, nullptr, nullptr, &err) == SQLITE_OK, "%s: %s", what, err ? err : "");
    sqlite3_free(err);
    sqlite3_close(db);
    Gallery t("test_v1", dim);
    t.upsert({"keep", {unit(dim, 0)}});
    CHECK_MSG(throwsKind([&] { t.load(path); }, GalleryError::Kind::Corrupt), "%s", what);
    CHECK_MSG(t.entries().size() == 1 && t.entries()[0].id == "keep", "%s", what);
  };
  corrupt("short blob", "UPDATE samples SET embedding = zeroblob(3) WHERE user_id = 'admin'");
  corrupt("text blob", "UPDATE samples SET embedding = 'abc' WHERE user_id = 'admin'");
  corrupt("user without samples", "INSERT INTO users(id) VALUES('ghost')");
  corrupt("sample gap", "UPDATE samples SET idx = 1 WHERE user_id = 'admin'");
  corrupt("no dim", "DELETE FROM meta WHERE key = 'dim'");
  corrupt("bad next_seq", "UPDATE meta SET value = 'x' WHERE key = 'next_seq'");
  corrupt("version 0", "PRAGMA user_version = 0");
  {  // a newer schema is not corruption: the file must not be moved away
    std::filesystem::remove(path);
    s.save(path);
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(path.c_str(), &db) == SQLITE_OK);
    CHECK(sqlite3_exec(db, "PRAGMA user_version = 99", nullptr, nullptr, nullptr) == SQLITE_OK);
    sqlite3_close(db);
    CHECK(throwsKind([&] { Gallery("test_v1", dim).load(path); }, GalleryError::Kind::Version));
    CHECK(throwsKind([&] { s.save(path); }, GalleryError::Kind::Version));
  }

  // files that are not a gallery database (the old PALMGAL format among them) are Corrupt and never overwritten
  for (const char* junk : {"PALMGAL3 old format", "not a gallery", "SQLite format 2"}) {
    { std::ofstream(path, std::ios::binary | std::ios::trunc) << junk; }
    Gallery t("test_v1", dim);
    t.upsert({"keep", {unit(dim, 0)}});
    CHECK_MSG(throwsKind([&] { t.load(path); }, GalleryError::Kind::Corrupt), "%s", junk);
    CHECK(t.entries().size() == 1);
    CHECK_MSG(throwsKind([&] { t.save(path); }, GalleryError::Kind::Corrupt), "%s", junk);
    std::ifstream in(path, std::ios::binary);
    std::string kept((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK_MSG(kept == junk, "%s overwritten", junk);
  }
  {  // a database cut short is Corrupt too
    std::filesystem::remove(path);
    s.save(path);
    std::filesystem::resize_file(path, 200);
    Gallery t("test_v1", dim);
    CHECK(throwsKind([&] { t.load(path); }, GalleryError::Kind::Corrupt));
  }

  // another connection holds the write lock: save waits, then reports Busy and the old content survives
  {
    std::filesystem::remove(path);
    s.save(path);
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(path.c_str(), &db) == SQLITE_OK);
    CHECK(sqlite3_exec(db, "BEGIN EXCLUSIVE", nullptr, nullptr, nullptr) == SQLITE_OK);
    Gallery t("test_v1", dim);
    t.setBusyTimeoutMs(100);
    t.upsert({"late", {unit(dim, 5)}});
    CHECK(throwsKind([&] { t.save(path); }, GalleryError::Kind::Busy));
    CHECK(throwsKind([&] { t.load(path); }, GalleryError::Kind::Busy));
    CHECK(sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr) == SQLITE_OK);
    sqlite3_close(db);
    Gallery u("test_v1", dim);
    u.load(path);
    CHECK(u.entries().size() == 3 && !u.contains("late"));
    t.save(path);  // free again
    u.load(path);
    CHECK(u.contains("late"));
  }
  std::filesystem::remove(path);

  return finish("test_gallery");
}
