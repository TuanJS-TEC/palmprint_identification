#include "gallery.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include "sqlite3.h"

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

TemplateResult buildTemplate(const std::vector<Vec>& feats, const TemplateParams& p) {
  TemplateResult r;
  const int n = static_cast<int>(feats.size());
  if (n == 0) {
    r.reason = "no_samples";
    return r;
  }
  // medoid: the sample with the highest mean cosine to the others
  std::vector<float> meanCos(n, 0.f);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i != j) meanCos[i] += dot(feats[i], feats[j]) / std::max(1, n - 1);
  r.medoid = static_cast<int>(std::max_element(meanCos.begin(), meanCos.end()) - meanCos.begin());

  Vec sum(feats[0].size(), 0.f);
  for (int i = 0; i < n; ++i) {
    const float c = dot(feats[i], feats[r.medoid]);
    r.cosToMedoid.push_back(c);
    if (c < p.outlierCosMin) continue;
    r.kept.push_back(i);
    for (size_t k = 0; k < sum.size(); ++k) sum[k] += feats[i][k];
  }
  if (static_cast<int>(r.kept.size()) < p.minValid) {
    r.reason = "too_few_valid_samples";
    return r;
  }
  r.tmpl = l2normalize(std::move(sum));
  r.ok = true;
  return r;
}

std::vector<Vec> keptSamples(const std::vector<Vec>& feats, const TemplateResult& t) {
  std::vector<Vec> out;
  for (int i : t.kept) out.push_back(feats.at(i));
  return out;
}

bool parseUserSeq(const std::string& id, uint32_t& n) {
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

// ---------------------------------------------------------------- storage
// SQLite database, schema version 1 (PRAGMA user_version):
//   meta(key, value)            model_version, dim, next_seq
//   users(id)                   in enrollment order (rowid)
//   samples(user_id, idx, embedding)   dim float32 (little-endian), idx = 0..n-1
// Each load / save opens its own connection, so a Gallery holds no handle.
// journal_mode stays DELETE: every commit changes the database file itself,
// which is what palm_api's file stamp watches (WAL would not).
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "embeddings are stored as IEEE float32");
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "embeddings are stored in host order, which must be little-endian"
#endif

constexpr uint32_t kMaxSamples = 1024;     // per entry; larger counts mean a corrupt database
constexpr size_t kMaxIdBytes = 1u << 20;   // longer ids mean a corrupt database
constexpr int kSchemaVersion = 1;
constexpr char kSqliteMagic[16] = {'S', 'Q', 'L', 'i', 't', 'e', ' ', 'f', 'o', 'r', 'm', 'a', 't', ' ', '3', '\0'};

using Kind = GalleryError::Kind;

void bumpSeq(uint32_t& seq, const std::string& id) {
  uint32_t n = 0;
  if (parseUserSeq(id, n)) seq = std::max(seq, n + 1);  // n <= 999999999: no overflow
}

struct DbCloser {
  void operator()(sqlite3* d) const { sqlite3_close_v2(d); }
};
struct StmtCloser {
  void operator()(sqlite3_stmt* s) const { sqlite3_finalize(s); }
};
using Db = std::unique_ptr<sqlite3, DbCloser>;
using Stmt = std::unique_ptr<sqlite3_stmt, StmtCloser>;

[[noreturn]] void fail(sqlite3* db, int rc, const std::string& what, const std::string& path) {
  Kind kind = Kind::Io;
  switch (rc & 0xff) {  // primary result code
    case SQLITE_BUSY:
    case SQLITE_LOCKED:
      kind = Kind::Busy;
      break;
    case SQLITE_NOTADB:
    case SQLITE_CORRUPT:
      kind = Kind::Corrupt;
      break;
    default:
      break;
  }
  throw GalleryError(kind, "gallery: " + what + " (" + path + "): " + (db ? sqlite3_errmsg(db) : sqlite3_errstr(rc)));
}

enum class FileState { Missing, Empty, Database, Other };

// What is at path, judged by the SQLite header: SQLite itself takes a short
// file of any content for an empty database, which would then be overwritten.
FileState probe(const std::string& path) {
  std::error_code ec;
  const auto st = std::filesystem::status(path, ec);
  if (st.type() == std::filesystem::file_type::not_found) return FileState::Missing;
  if (ec || st.type() != std::filesystem::file_type::regular)
    throw GalleryError(Kind::Io, "gallery: not a readable file: " + path);
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) throw GalleryError(Kind::Io, "gallery: cannot stat " + path);
  if (size == 0) return FileState::Empty;
  std::ifstream f(path, std::ios::binary);
  if (!f) throw GalleryError(Kind::Io, "gallery: cannot open " + path);
  char head[sizeof kSqliteMagic] = {};
  f.read(head, sizeof head);
  return f && std::equal(head, head + sizeof head, kSqliteMagic) ? FileState::Database : FileState::Other;
}

Db openDb(const std::string& path, int flags, int busyMs) {
  sqlite3* raw = nullptr;
  const int rc = sqlite3_open_v2(path.c_str(), &raw, flags | SQLITE_OPEN_NOMUTEX, nullptr);
  Db db(raw);
  if (rc != SQLITE_OK) fail(raw, rc, "cannot open", path);
  sqlite3_busy_timeout(raw, busyMs);
  return db;
}

void exec(sqlite3* db, const char* sql, const std::string& path) {
  const int rc = sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
  if (rc != SQLITE_OK) fail(db, rc, std::string("cannot run '") + sql + "'", path);
}

Stmt prepare(sqlite3* db, const char* sql, const std::string& path) {
  sqlite3_stmt* raw = nullptr;
  const int rc = sqlite3_prepare_v2(db, sql, -1, &raw, nullptr);
  Stmt s(raw);
  if (rc != SQLITE_OK) fail(db, rc, std::string("cannot prepare '") + sql + "'", path);
  return s;
}

int userVersion(sqlite3* db, const std::string& path) {
  Stmt s = prepare(db, "PRAGMA user_version", path);
  const int rc = sqlite3_step(s.get());
  if (rc != SQLITE_ROW) fail(db, rc, "cannot read user_version", path);
  return sqlite3_column_int(s.get(), 0);
}

bool parseUint(const std::string& text, uint32_t& out) {
  const char* last = text.data() + text.size();
  const auto r = std::from_chars(text.data(), last, out);
  return r.ec == std::errc() && r.ptr == last && !text.empty();
}

std::string columnText(sqlite3_stmt* s, int col) {
  const unsigned char* t = sqlite3_column_text(s, col);
  return t ? std::string(reinterpret_cast<const char*>(t), static_cast<size_t>(sqlite3_column_bytes(s, col))) : std::string();
}
}  // namespace

void Gallery::load(const std::string& path) {
  const FileState state = probe(path);
  if (state == FileState::Other) throw GalleryError(Kind::Corrupt, "gallery: not a palm gallery database: " + path);
  if (state != FileState::Database) {  // missing or empty: nothing enrolled yet, and no file is created
    entries_.clear();
    nextSeq_ = 1;
    return;
  }
  // READWRITE lets SQLite roll back a journal that a crashed save left behind;
  // it still opens read-only a file the process may not write.
  const Db db = openDb(path, SQLITE_OPEN_READWRITE, busyTimeoutMs_);
  sqlite3* d = db.get();
  exec(d, "BEGIN", path);  // one snapshot for all the queries below

  // parsed into locals: a corrupt database leaves the gallery as it was
  const int version = userVersion(d, path);
  if (version > kSchemaVersion)
    throw GalleryError(Kind::Version, "gallery: " + path + " has schema version " + std::to_string(version) +
                                          ", this program reads up to " + std::to_string(kSchemaVersion));
  std::vector<GalleryEntry> entries;
  uint32_t seq = 1;
  {
    Stmt has = prepare(d, "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'meta'", path);
    if (sqlite3_step(has.get()) != SQLITE_ROW) fail(d, sqlite3_errcode(d), "cannot read the schema", path);
    if (sqlite3_column_int(has.get(), 0) == 0) {  // a database file without a gallery in it
      entries_.clear();
      nextSeq_ = 1;
      return;
    }
  }
  if (version != kSchemaVersion) throw GalleryError(Kind::Corrupt, "gallery: no schema version in " + path);

  std::string modelVersion;
  uint32_t dim = 0;
  bool haveModel = false, haveDim = false, haveSeq = false;
  {
    Stmt meta = prepare(d, "SELECT key, value FROM meta", path);
    int rc;
    while ((rc = sqlite3_step(meta.get())) == SQLITE_ROW) {
      const std::string key = columnText(meta.get(), 0), value = columnText(meta.get(), 1);
      if (key == "model_version") {
        modelVersion = value;
        haveModel = true;
      } else if (key == "dim") {
        haveDim = parseUint(value, dim);
      } else if (key == "next_seq") {
        haveSeq = parseUint(value, seq);
      }
    }
    if (rc != SQLITE_DONE) fail(d, rc, "cannot read meta", path);
  }
  if (!haveModel || !haveDim || !haveSeq || seq == 0 || dim == 0 || dim > (1u << 20))
    throw GalleryError(Kind::Corrupt, "gallery: incomplete meta table in " + path);
  if (modelVersion != modelVersion_ || static_cast<int>(dim) != dim_)
    throw GalleryError(Kind::ModelMismatch, "gallery: built with model '" + modelVersion + "' (dim " +
                                                std::to_string(dim) + "), current model is '" + modelVersion_ +
                                                "'; re-enroll");

  {
    Stmt rows = prepare(d,
                        "SELECT u.id, s.idx, s.embedding FROM users u LEFT JOIN samples s ON s.user_id = u.id "
                        "ORDER BY u.rowid, s.idx",
                        path);
    const size_t bytes = static_cast<size_t>(dim_) * sizeof(float);
    int rc;
    while ((rc = sqlite3_step(rows.get())) == SQLITE_ROW) {
      std::string id = columnText(rows.get(), 0);
      if (id.size() > kMaxIdBytes) throw GalleryError(Kind::Corrupt, "gallery: corrupt id length: " + path);
      if (entries.empty() || entries.back().id != id) {
        if (!entries.empty() && entries.back().samples.empty())
          throw GalleryError(Kind::Corrupt, "gallery: '" + entries.back().id + "' has no samples: " + path);
        entries.push_back({std::move(id), {}});
      }
      GalleryEntry& e = entries.back();
      if (sqlite3_column_type(rows.get(), 1) == SQLITE_NULL) continue;  // a user without samples: checked at the next user / the end
      if (sqlite3_column_type(rows.get(), 2) != SQLITE_BLOB || static_cast<size_t>(sqlite3_column_bytes(rows.get(), 2)) != bytes ||
          sqlite3_column_int64(rows.get(), 1) != static_cast<sqlite3_int64>(e.samples.size()) ||
          e.samples.size() >= kMaxSamples)
        throw GalleryError(Kind::Corrupt, "gallery: corrupt sample of '" + e.id + "': " + path);
      Vec v(dim_);
      std::memcpy(v.data(), sqlite3_column_blob(rows.get(), 2), bytes);
      e.samples.push_back(std::move(v));
    }
    if (rc != SQLITE_DONE) fail(d, rc, "cannot read the users", path);
    if (!entries.empty() && entries.back().samples.empty())
      throw GalleryError(Kind::Corrupt, "gallery: '" + entries.back().id + "' has no samples: " + path);
  }
  for (const GalleryEntry& e : entries) bumpSeq(seq, e.id);
  entries_ = std::move(entries);
  nextSeq_ = seq;
}

void Gallery::save(const std::string& path) const {
  if (probe(path) == FileState::Other)
    throw GalleryError(Kind::Corrupt, "gallery: refusing to overwrite a file that is not a palm gallery: " + path);
  const Db db = openDb(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, busyTimeoutMs_);
  sqlite3* d = db.get();
  exec(d, "PRAGMA journal_mode = DELETE", path);
  exec(d, "PRAGMA synchronous = FULL", path);
  exec(d, "PRAGMA foreign_keys = ON", path);
  exec(d, "BEGIN IMMEDIATE", path);  // the write lock is taken here; Busy after the timeout
  try {
    if (userVersion(d, path) > kSchemaVersion)
      throw GalleryError(Kind::Version, "gallery: " + path + " was written by a newer program");
    exec(d,
         "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
         "CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY);"
         "CREATE TABLE IF NOT EXISTS samples(user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,"
         " idx INTEGER NOT NULL, embedding BLOB NOT NULL, PRIMARY KEY(user_id, idx)) WITHOUT ROWID;"
         "DELETE FROM samples; DELETE FROM users; DELETE FROM meta;",
         path);
    {
      Stmt meta = prepare(d, "INSERT INTO meta(key, value) VALUES(?, ?)", path);
      const auto put = [&](const char* key, const std::string& value) {
        sqlite3_reset(meta.get());
        sqlite3_bind_text(meta.get(), 1, key, -1, SQLITE_STATIC);
        sqlite3_bind_text(meta.get(), 2, value.c_str(), static_cast<int>(value.size()), SQLITE_STATIC);
        if (const int rc = sqlite3_step(meta.get()); rc != SQLITE_DONE) fail(d, rc, "cannot write meta", path);
      };
      put("model_version", modelVersion_);
      put("dim", std::to_string(dim_));
      put("next_seq", std::to_string(nextSeq_));
    }
    Stmt user = prepare(d, "INSERT INTO users(id) VALUES(?)", path);
    Stmt sample = prepare(d, "INSERT INTO samples(user_id, idx, embedding) VALUES(?, ?, ?)", path);
    const int bytes = static_cast<int>(static_cast<size_t>(dim_) * sizeof(float));
    for (const auto& e : entries_) {
      sqlite3_reset(user.get());
      sqlite3_bind_text(user.get(), 1, e.id.c_str(), static_cast<int>(e.id.size()), SQLITE_STATIC);
      if (const int rc = sqlite3_step(user.get()); rc != SQLITE_DONE) fail(d, rc, "cannot write user '" + e.id + "'", path);
      for (size_t i = 0; i < e.samples.size(); ++i) {
        sqlite3_reset(sample.get());
        sqlite3_bind_text(sample.get(), 1, e.id.c_str(), static_cast<int>(e.id.size()), SQLITE_STATIC);
        sqlite3_bind_int64(sample.get(), 2, static_cast<sqlite3_int64>(i));
        sqlite3_bind_blob(sample.get(), 3, e.samples[i].data(), bytes, SQLITE_STATIC);
        if (const int rc = sqlite3_step(sample.get()); rc != SQLITE_DONE)
          fail(d, rc, "cannot write a sample of '" + e.id + "'", path);
      }
    }
    exec(d, ("PRAGMA user_version = " + std::to_string(kSchemaVersion)).c_str(), path);
    exec(d, "COMMIT", path);
  } catch (...) {
    sqlite3_exec(d, "ROLLBACK", nullptr, nullptr, nullptr);
    throw;
  }
}

bool Gallery::contains(const std::string& id) const {
  return std::any_of(entries_.begin(), entries_.end(), [&](const GalleryEntry& e) { return e.id == id; });
}

std::string Gallery::nextId() const {
  char id[24];
  std::snprintf(id, sizeof id, "user_%04u", static_cast<unsigned>(nextSeq_));
  return id;
}

void Gallery::upsert(GalleryEntry e) {
  if (e.samples.empty() || e.samples.size() > kMaxSamples)
    throw std::invalid_argument("gallery: an entry needs 1.." + std::to_string(kMaxSamples) + " samples");
  for (const Vec& v : e.samples)
    if (static_cast<int>(v.size()) != dim_) throw std::invalid_argument("gallery: sample dimension mismatch");
  bumpSeq(nextSeq_, e.id);
  for (auto& x : entries_)
    if (x.id == e.id) {
      x = std::move(e);
      return;
    }
  entries_.push_back(std::move(e));
}

bool Gallery::remove(const std::string& id) {
  const auto it = std::remove_if(entries_.begin(), entries_.end(), [&](const GalleryEntry& e) { return e.id == id; });
  const bool found = it != entries_.end();
  entries_.erase(it, entries_.end());
  return found;
}

float bestScore(const Vec& probe, const GalleryEntry& e) {
  float best = -2.f;  // below any cosine
  for (const Vec& v : e.samples) best = std::max(best, dot(probe, v));
  return best;
}

namespace {
void sortTop(std::vector<Match>& m, int topK) {
  std::sort(m.begin(), m.end(), [](const Match& a, const Match& b) { return a.score > b.score; });
  if (static_cast<int>(m.size()) > topK) m.resize(std::max(0, topK));
}
}  // namespace

std::vector<Match> Gallery::search(const Vec& probe, int topK, const std::string& excludeId) const {
  std::vector<Match> m;
  for (const auto& e : entries_)
    if (e.id != excludeId) m.push_back({e.id, bestScore(probe, e)});
  sortTop(m, topK);
  return m;
}

std::vector<Match> Gallery::search(const std::vector<Vec>& probes, int topK, const std::string& excludeId) const {
  std::vector<Match> m;
  for (const auto& e : entries_) {
    if (e.id == excludeId) continue;
    float best = -2.f;
    for (const Vec& p : probes) best = std::max(best, bestScore(p, e));
    m.push_back({e.id, best});
  }
  sortTop(m, topK);
  return m;
}

}  // namespace palm
