#include <cmath>
#include <filesystem>

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
  g.upsert({"a", 3, unit(dim, 0)});
  g.upsert({"b", 3, unit(dim, 1)});
  g.upsert({"a", 4, unit(dim, 2)});  // replaces
  CHECK(g.entries().size() == 2);
  auto m = g.search(unit(dim, 2, 0.05f), 2);
  CHECK(m.size() == 2 && m[0].id == "a" && m[0].score > m[1].score);
  CHECK(g.search(unit(dim, 2), 5, "a").size() == 1);

  const auto path = (std::filesystem::temp_directory_path() / "palm_test_gallery.bin").string();
  g.save(path);
  Gallery h("test_v1", dim);
  h.load(path);
  CHECK(h.entries().size() == 2 && h.entries()[0].id == "a" && h.entries()[0].numSamples == 4);
  CHECK(h.entries()[1].tmpl == g.entries()[1].tmpl);
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

  return finish("test_gallery");
}
