#include "palm/palm_model.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

#include "preprocess.h"

namespace palm {
namespace {

const char* kInputs[] = {"input_56", "input_224"};
const char* kOutputs[] = {"feature", "theta", "roi"};  // "logits" (NTU-PI classes) is not requested

std::vector<std::string> names(Ort::Session& s, bool inputs) {
  Ort::AllocatorWithDefaultOptions alloc;
  std::vector<std::string> v;
  const size_t n = inputs ? s.GetInputCount() : s.GetOutputCount();
  for (size_t i = 0; i < n; ++i)
    v.emplace_back((inputs ? s.GetInputNameAllocated(i, alloc) : s.GetOutputNameAllocated(i, alloc)).get());
  return v;
}

void require(const std::vector<std::string>& have, const char* const* want, size_t n, const std::string& what) {
  for (size_t i = 0; i < n; ++i)
    if (std::find(have.begin(), have.end(), want[i]) == have.end())
      throw std::runtime_error("ONNX model has no " + what + " '" + want[i] +
                               "' (expected EE_PRnet-epoch-60.onnx from convert_mat_to_onnx.py)");
}

}  // namespace

struct PalmModel::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "palm"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  int featureDim = 0;

  explicit Impl(const ModelOptions& opt) {
    if (!std::filesystem::exists(opt.onnxPath)) throw std::runtime_error("model not found: " + opt.onnxPath);
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(opt.threads);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");  // several sessions share the CPU
    session = std::make_unique<Ort::Session>(env, std::filesystem::path(opt.onnxPath).c_str(), so);
    require(names(*session, true), kInputs, 2, "input");
    require(names(*session, false), kOutputs, 3, "output");
    for (size_t i = 0; i < session->GetOutputCount(); ++i) {
      Ort::AllocatorWithDefaultOptions alloc;
      if (std::string(session->GetOutputNameAllocated(i, alloc).get()) == "feature")
        featureDim = static_cast<int>(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape().back());
    }
  }
};

PalmModel::PalmModel(const ModelOptions& opt) : p_(std::make_unique<Impl>(opt)) {}
PalmModel::~PalmModel() = default;

int PalmModel::featureDim() const { return p_->featureDim; }

Embedding PalmModel::run(const cv::Mat& bgr) {
  const cv::Mat rgb = toRgbFloat(bgr);
  const cv::Mat x56 = vlResize(rgb, {56, 56}), x224 = vlResize(rgb, {224, 224});
  std::vector<float> a(3 * x56.total()), b(3 * x224.total());
  packChw(x56, a.data());
  packChw(x224, b.data());
  const std::array<int64_t, 4> sa{1, 3, x56.rows, x56.cols}, sb{1, 3, x224.rows, x224.cols};
  std::array<Ort::Value, 2> x{
      Ort::Value::CreateTensor<float>(p_->mem, a.data(), a.size(), sa.data(), sa.size()),
      Ort::Value::CreateTensor<float>(p_->mem, b.data(), b.size(), sb.data(), sb.size())};
  auto y = p_->session->Run(Ort::RunOptions{nullptr}, kInputs, x.data(), x.size(), kOutputs, 3);

  Embedding e;
  const float* f = y[0].GetTensorData<float>();
  e.feature.assign(f, f + y[0].GetTensorTypeAndShapeInfo().GetElementCount());
  const float* t = y[1].GetTensorData<float>();
  std::copy(t, t + 18, e.theta.begin());
  const float* r = y[2].GetTensorData<float>();  // 1x3x112x112
  const auto shape = y[2].GetTensorTypeAndShapeInfo().GetShape();
  const int h = static_cast<int>(shape[2]), w = static_cast<int>(shape[3]);
  std::vector<cv::Mat> planes;
  for (int c = 0; c < 3; ++c) planes.emplace_back(h, w, CV_32F, const_cast<float*>(r + c * h * w));
  cv::merge(planes, e.roi);  // copies
  return e;
}

}  // namespace palm
