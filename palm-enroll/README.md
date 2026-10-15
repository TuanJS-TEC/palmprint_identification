# palm-enroll

Lõi C++ nhận dạng lòng bàn tay phải: trích đặc trưng 512 chiều bằng EE-PRnet (Matkowski et al., IEEE TIFS 2019), lưu từng mẫu đăng ký vào gallery SQLite, nhận dạng 1:N bằng cosine. Gồm thư viện `palm` và công cụ dòng lệnh `palm_cli`.

Model: `../models/EE_PRnet-epoch-60.onnx` (tạo bằng `../Transform Model/convert_mat_to_onnx.py`) và hai model MediaPipe phát hiện tay (OpenCV Zoo, Apache-2.0).

Phụ thuộc: OpenCV 4 (core, imgproc, imgcodecs, videoio), ONNX Runtime ≥ 1.16 (CPU), C++17, CMake ≥ 3.18.

## Bố cục

```
include/palm/    API công khai (palm/palm.h include tất cả)
src/             cài đặt, chia module: common, recognition, detection, gallery, config
apps/palm_cli/   công cụ dòng lệnh
docs/            architecture.md, conventions.md, measurements.md
config.yaml      cấu hình của lõi
```

Chi tiết và quy tắc phụ thuộc: [docs/architecture.md](docs/architecture.md).

## Build

```bash
cmake -S . -B build -G Ninja          # ORT khác /usr/local: thêm -DORT_ROOT=<thư mục onnxruntime>
cmake --build build
cmake --install build --prefix ../binary   # palm_cli + config.yaml + models vào ../binary
```

Trên Windows: `-DORT_ROOT=C:/onnxruntime-win-x64-<ver> -DOpenCV_DIR=<...>/build`.

Dùng làm thư viện trong dự án khác:

```cmake
add_subdirectory(../palm-enroll palm)          # không build palm_cli khi là dự án con
target_link_libraries(my_app PRIVATE palm::palm)
```
```cpp
#include <palm/palm.h>
```

## Sử dụng `palm_cli`

Mỗi người chỉ đăng ký **một tay, là tay phải**. Ảnh đưa vào phải là crop cả bàn tay phải nhìn thẳng, ngón cái bên phải ảnh; `palm_cli` không kiểm tra ảnh là tay nào (với khung camera thì dùng `HandDetector` và `CaptureGate`, xem `capture_gate.h`).

```bash
./build/palm_cli enroll   --id P01 img1.jpg ...      # >= template.min_valid mẫu tốt
./build/palm_cli identify img.jpg                    # top-3 + quyết định theo accept_cos_min
./build/palm_cli embed    --out-dir out img.jpg      # .f32 (512 float) + ROI 112x112 .png
./build/palm_cli list | remove --id P01
```

`enroll` từ chối khi: ID đã đăng ký (dùng `--force` để thay), hoặc mẫu khớp một người khác trên `dup_cos_max`, tức cùng bàn tay đăng ký dưới hai tên. Mọi lệnh nhận `--config` (mặc định `config.yaml` trong thư mục hiện tại) và `--gallery PATH` để dùng gallery khác với gallery trong config.

## Tài liệu

- [docs/architecture.md](docs/architecture.md): module, phụ thuộc, luồng dữ liệu, định dạng gallery.
- [docs/conventions.md](docs/conventions.md): quy ước đã chốt (tiền xử lý, hướng ảnh, khung cắt) và khác biệt so với tài liệu hướng dẫn.
- [docs/measurements.md](docs/measurements.md): số đo gốc của các ngưỡng, ảnh hưởng của góc nghiêng.

## Chưa làm

- Mã hóa gallery (`gallery.db` hiện **chưa mã hóa**).
- Liveness.
- Đưa phiên đăng ký / nhận dạng vào lõi (hiện ứng dụng nào dùng cũng phải tự viết).
- Test tự động cho lõi (đã bị gỡ ở commit `eb5e67d`).
- Đo lại `accept_cos_min` / `dup_cos_max` cho cách khớp theo mẫu, trên dữ liệu đăng ký đa góc.
- Đo `max_motion` và `min_crop_frac` trên camera thật: hiện mới là giá trị khởi đầu.
