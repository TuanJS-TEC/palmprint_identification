# palm-enroll

Pipeline C++ trích đặc trưng lòng bàn tay 512 chiều (EE-PRnet, Matkowski et al., IEEE TIFS 2019) từ ảnh crop bàn tay, gộp nhiều mẫu thành template và nhận dạng 1:N. Có giao diện Qt (`palm_gui`) dùng camera thật: phát hiện tay, đăng ký tay phải và nhận dạng. Model: `../models/EE_PRnet-epoch-60.onnx`, do `../convert_mat_to_onnx.py` tạo, cùng hai model MediaPipe phát hiện tay (OpenCV Zoo, Apache-2.0).

Phụ thuộc: OpenCV 4 (core, imgproc, imgcodecs, videoio), ONNX Runtime ≥ 1.16 (CPU), C++17, CMake ≥ 3.18. Qt 5 Widgets là tùy chọn: thiếu thì không build `palm_gui`.

## Giao diện với camera (`palm_gui`)

```bash
cmake --install build --prefix ../binary   # chương trình + config + models vào ../binary
../binary/run_gui.sh                       # màn hình gắn vào máy (EGLFS), hoặc cửa sổ nếu có X
../binary/run_gui.sh --vnc                 # xem từ xa qua VNC, cổng 5900
```

Hướng dẫn cho người test nằm trong `binary/README.txt`. Mỗi khung hình đi qua: phát hiện tay (palm detector + 21 landmark), kiểm tra tay phải và lòng bàn tay hướng camera, kích thước, nằm trọn trong ảnh, độ rung, độ nét, rồi cắt khung bàn tay và chạy EE-PRnet. Tham số thử nghiệm:
- `--source ảnh|video [--mirrored]`: dùng ảnh hoặc video thay camera.
- `--enroll ID` / `--identify`: vào thẳng chế độ đăng ký hoặc nhận dạng.
- `--verbose`: in thời gian xử lý mỗi giây.

Trên máy dev, chạy liên tục có tay: phát hiện tay 22 ms, embedding 32 ms, khoảng 18 fps. Không có tay: khoảng 26 fps.

## Build và test

```bash
cmake -S . -B build -G Ninja          # ORT khác /usr/local: thêm -DORT_ROOT=<thư mục onnxruntime>
cmake --build build
ctest --test-dir build --output-on-failure
```

Trên Windows: `-DORT_ROOT=C:/onnxruntime-win-x64-<ver> -DOpenCV_DIR=<...>/build`.

Test `parity` so C++ với tham chiếu Python trong `tests/data/ref/`. Chạy lại `../venv/bin/python python/export_cpp_ref.py` khi model hoặc tiền xử lý thay đổi.

## Sử dụng

Mỗi người chỉ đăng ký **một tay, là tay phải**. `enroll` và `identify` bắt buộc có `--label`, là nhãn handedness của MediaPipe (cột `handedness` trong `log_label.csv`). Nhãn không ứng với tay phải sẽ bị từ chối ngay, trước khi chạy model.

```bash
# đo trên các ảnh tay phải của tập có nhãn, in ngưỡng gợi ý cho config.yaml
./build/palm_eval --labels ../log_label.csv --images ../output/hand_crop \
                  [--enroll 5] [--scores s.csv] [--roi-sheet rois.jpg]

./build/palm_cli enroll   --id P01 --label Left img1.jpg ...  # >= template.min_valid mẫu tốt
./build/palm_cli identify --label Left img.jpg                # top-3 + quyết định theo accept_cos_min
./build/palm_cli embed    --out-dir out img.jpg               # .f32 (512 float) + ROI 112x112 .png
./build/palm_cli list | remove --id P01

# kiểm tra bộ phát hiện tay trên ảnh gốc có nhãn (nhãn, phía lòng bàn tay, landmark ra CSV)
./build/palm_handcheck --labels ../log_label.csv --images ../output/original --mirrored [--out hands.csv]
```

`enroll` từ chối khi: ID đã đăng ký (dùng `--force` để thay template), hoặc template khớp một người khác trên `dup_cos_max`, tức cùng bàn tay đăng ký dưới hai tên. Mọi lệnh nhận `--config`, mặc định là `config.yaml` trong thư mục hiện tại.

## Quy ước đã chốt

| Mục | Giá trị | Nguồn |
| --- | --- | --- |
| Ảnh vào | Crop cả bàn tay (như `output/hand_crop`), RGB 0–255; trừ mean ImageNet nằm trong ONNX | `buildFeat.m` |
| Resize 56×56 và 224×224 | Như `vl_imreadjpeg`: crop giữa theo tỉ lệ khung đầu ra, lọc tam giác float nở theo tỉ lệ thu nhỏ, biên replicate. **Không** dùng `cv::resize` | `vl_imreadjpeg.cu`, `imread_helpers.hpp` |
| `input_224` | Luôn 224×224 như lúc train, kể cả checkpoint ghi `imgOrg` | `getBatchNclassSpecial.m` |
| θ (18 số) | `[x1..x9, y1..y9]`, tọa độ tuyệt đối trong [-1, 1], 9 điểm trên lưới 3×3 thứ tự cột của MATLAB | `TpsGridGenerator.m` |
| Lấy mẫu TPS | `align_corners=True`, ngoài ảnh = 0 sau khi trừ mean; nằm trong ONNX, C++ không cần `tps_T.bin` | `vl_nnbilinearsampler` |
| Tay nào là tay phải (trong `log_label.csv`) | Nhãn **"Left"** (`hand.right_label`). Script thu ảnh lật gương khung hình trước khi chạy MediaPipe, mà MediaPipe gán nhãn theo hình dạng tay trong ảnh nó nhận | Chữ ngược trên áo trong `output/original`, đồng hồ, `palm_handcheck`: 405/408 ảnh nhãn Left có lòng bàn tay phải hướng camera |
| Tay phải trong `palm_gui` | Khung camera thật (không lật): nhãn detector **"Right"** và tích chéo (ngón trỏ, ngón út quanh cổ tay) < 0. Nhãn và hình học khớp nhau ở 97.6% ảnh | `palm_handcheck` |
| Hướng ảnh | Lòng bàn tay phải nhìn thẳng, ngón cái bên phải ảnh, như NTU-PI-v1. `output/hand_crop` đã đúng hướng (script thu ảnh lật lại crop của nhãn Left), nên `mirror_input: 0` | `palm_eval` |
| Khung cắt bàn tay | Hình vuông tâm trùng tâm khung bao 21 landmark, cạnh 1.17 × max(rộng, cao). Hiệu chỉnh trên 596 crop (p10–p90: 1.14–1.20) | so khớp `hand_crop` với ảnh gốc |

## Kết quả đo (2026-10-05, `eval/report.txt`)

409 ảnh tay phải trong `output/hand_crop` của 25 người. Mỗi người đăng ký bằng 5 mẫu đầu theo thời gian, các mẫu còn lại làm probe.

| Chỉ số | Giá trị |
| --- | --- |
| Rank-1 (25 người, 284 probe) | 89.1% |
| EER probe–template | 9.2% |
| Ngưỡng 0.82 | 0 chấp nhận nhầm, 16.5% ảnh đơn bị từ chối |
| Suy luận / tiền xử lý (máy dev, 4 luồng) | 29 ms / 6 ms mỗi ảnh |
| C++ so với Python ORT | Trùng bit (sai lệch 0) |

Lật ảnh tay phải làm kết quả kém đi: EER 10.6%, rank-1 86.6%. Ngưỡng được chọn trên chính tập này và FAR 0.1% chỉ dựa trên khoảng 6 điểm impostor, nên số thật có thể kém hơn.

Đo đúng đường xử lý của `palm_gui` trên ảnh gốc (`palm_eval --frames ../output/original --frames-mirrored`):
- 324/410 ảnh qua được các bước kiểm tra; 81 ảnh bị loại vì bàn tay chạm mép ảnh.
- Kết quả: rank-1 88.8%, EER 9.8%; tại ngưỡng 0.82 có 84.2% chấp nhận đúng, 0 chấp nhận nhầm.
- Như vậy tương đương khi dùng `hand_crop`.

Lưu ý:
- Năm mẫu chụp liên tiếp trong vài giây gần như trùng nhau (cos > 0.99), nên template ít đa dạng. `palm_gui` chụp cách nhau 0.7 s và nhắc nghiêng tay giữa các lần chụp.
- Có ROI hỏng: người 22 bị mờ, người 24 trông như da mặt (xem `eval/rois.jpg`). `palm_gui` loại ảnh mờ (`blur_min`), ảnh rung và bàn tay nằm ngoài khung.

## Khác với tài liệu hướng dẫn

- Dùng một ONNX duy nhất thay cho `theta.onnx`, `embed.onnx` và `tps_T.bin`, vì TPS và sampler đã nằm trong graph.
- Không dùng yaml-cpp, Catch2, Eigen, SQLite hay libsodium (máy chưa cài). Thay vào đó: `cv::FileStorage` cho YAML, test tự viết chạy bằng CTest, `std::vector` cho vector đặc trưng.
- `gallery.bin` là file nhị phân **chưa mã hóa**. Phải thêm mã hóa (libsodium) trước khi chạy với dữ liệu thật.

## Chưa làm

- Mã hóa gallery.
- Liveness.
- Buộc các mẫu đăng ký phải khác tư thế nhau: hiện chỉ có nhắc nhở và khoảng cách thời gian giữa hai lần chụp.
- Đo các ngưỡng `max_motion` và `min_crop_frac` trên camera thật: hiện mới là giá trị khởi đầu.
