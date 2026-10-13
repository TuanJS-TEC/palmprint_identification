# palmprint_identification

Nhận dạng lòng bàn tay từ camera: model EE-PRnet (Matkowski et al., IEEE TIFS 2019) chuyển từ MatConvNet sang ONNX, pipeline C++ đăng ký tay phải và nhận dạng 1:N, giao diện Qt chạy trên máy có camera và SDK (C API) cho nhóm mobile. Chi tiết build, sử dụng và kết quả đo nằm trong [palm-enroll/README.md](palm-enroll/README.md).

## Cấu trúc thư mục

```text
palmprint_identification/
├── palm-enroll/            dự án C++ (CMake)
│   ├── src/                thư viện palm: tiền xử lý, EE-PRnet, phát hiện tay, gallery, C API
│   ├── gui/                palm_gui (Qt 5, camera)
│   ├── tools/              palm_cli, palm_eval, palm_bench, palm_profile, palm_soak, palm_handcheck
│   ├── tests/              ctest; tests/data/ref là tham chiếu Python cho test parity
│   ├── sdk/                đóng gói SDK cho mobile (make_sdk.sh)
│   ├── deploy/             file đi kèm khi cmake --install vào binary/
│   ├── eval/               báo cáo đo ngưỡng và hiệu năng
│   ├── python/             script Python gắn với mã C++ (tham chiếu, profile, biểu đồ)
│   └── config.yaml         ngưỡng, đường dẫn model và gallery
├── scripts/
│   ├── convert_mat_to_onnx.py   .mat (MatConvNet) -> .onnx
│   └── verify_onnx.py           kiểm tra ONNX với bản NumPy độc lập
├── data/log_label.csv      log thu ảnh: nhãn tay, người, ROI của từng ảnh trong output/
└── docs/cpp-enroll-guide.md     hướng dẫn thiết kế ban đầu (5/10/2026)
```

Các thư mục sau chỉ có trên máy, không đưa vào git (xem `.gitignore`):

| Thư mục | Nội dung | Lấy từ đâu |
| --- | --- | --- |
| `models/` | `*.mat`, `*.onnx` của EE-PRnet/AlignNet, 2 model MediaPipe phát hiện tay | `.mat` từ github.com/matkowski-voy/Palmprint-Recognition-in-the-Wild, chuyển bằng `scripts/convert_mat_to_onnx.py`; model MediaPipe từ OpenCV Zoo (Apache-2.0) |
| `output/` | ảnh thu từ camera: `original/` (khung hình lật gương), `hand_crop/`, ... | script thu ảnh, khớp với `data/log_label.csv` |
| `Palmprint-Recognition-in-the-Wild-master/` | mã gốc của bài báo và tập NTU-PI-demo | repo trên, dùng bởi `verify_onnx.py` và `export_cpp_ref.py` |
| `venv/` | môi trường Python (phiên bản trong `.python-version`) | `python -m venv venv` |
| `binary/` | bản chạy độc lập: chương trình, config, models, gallery | `cmake --install build --prefix ../binary` trong `palm-enroll/` |
| `dist/` | gói SDK | `palm-enroll/sdk/make_sdk.sh` |

## Bắt đầu nhanh

```bash
venv/bin/python scripts/convert_mat_to_onnx.py   # models/*.mat -> models/*.onnx
venv/bin/python scripts/verify_onnx.py            # tùy chọn: kiểm tra model đã chuyển

cd palm-enroll
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix ../binary
../binary/run_gui.sh
```
