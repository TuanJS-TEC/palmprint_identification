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
- `--log FILE`: file log sự kiện, mặc định `palm_gui.log` cạnh `config.yaml`; `--log ""` chỉ hiện trên màn hình.

Khung log dưới video (F2 ẩn/hiện) ghi: trạng thái tay khi đổi kèm độ tin cậy của hai model MediaPipe (`palm` điểm phát hiện lòng bàn tay, `pres` có bàn tay, `right` xác suất tay phải, độ nét), từng mẫu đăng ký (cos với các mẫu trước, người khác gần nhất), từng lần nhận dạng (ID gần nhất và cos, người thứ hai và khoảng cách, đạt/không đạt ngưỡng), kết quả đăng ký/nhận dạng và thời gian xử lý mỗi 30 s. EE-PRnet không cho xác suất, nên độ tin cậy của nó là cos: trên đường xử lý camera, ảnh đúng người có trung vị 0.95, người khác trung vị 0.55 và 99% dưới 0.76 (`eval/report_live_path.txt`). Log được ghi cả vào file (xoay vòng sang `.1` ở 10 MB).

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

# qua C API, trên khung hình camera đầy đủ (--mirrored cho ảnh lật gương như output/original)
./build/palm_cli register [--mirrored] f1.jpg ... f5.jpg      # -> OK user_0003 | ALREADY_REGISTERED user_0001
./build/palm_cli check    [--mirrored] frame.jpg              # -> user_0001 | palm not recognised

# kiểm tra bộ phát hiện tay trên ảnh gốc có nhãn (nhãn, phía lòng bàn tay, landmark ra CSV)
./build/palm_handcheck --labels ../log_label.csv --images ../output/original --mirrored [--out hands.csv]
```

`enroll` từ chối khi: ID đã đăng ký (dùng `--force` để thay template), hoặc template khớp một người khác trên `dup_cos_max`, tức cùng bàn tay đăng ký dưới hai tên. Mọi lệnh nhận `--config`, mặc định là `config.yaml` trong thư mục hiện tại, và `--gallery PATH` để dùng gallery khác với gallery trong config.

## API C (`src/palm_api.h`)

Dùng cho ứng dụng khác gọi vào, ví dụ app Android qua JNI. Đầu vào là byte của khung hình camera; phát hiện tay, kiểm tra tay phải và crop đều làm bên trong. Các hàm chính:

| Hàm | Việc |
| --- | --- |
| `palm_init(&opt, &h)` / `palm_release(h)` | Nạp 3 model ONNX và gallery. `opt.config_path` là `config.yaml`, hoặc NULL kèm 3 đường dẫn model và `gallery_path` (khi đó dùng ngưỡng đo ngày 2026-10-05) |
| `palm_extract(h, &frame, feat, 512)` | Khung hình → vector 512 chiều đã chuẩn hóa L2 |
| `palm_compare(h, a, b, 512, &score, &same)` | Cosine của 2 vector, `same = score >= accept_cos_min` |
| `palm_register(h, frames, n, id, 64, &score, status)` | **Register**: đăng ký người mới, trả ID `user_0001`, `user_0002`, ... |
| `palm_check_register(h, &frame, id, 64, &score)` | **Check_Register**: nhận dạng 1:N, trả ID hoặc `PALM_ERR_NOT_RECOGNIZED` |

```c
PalmInitOptions opt = {0};
opt.config_path = "config.yaml";
PalmHandle h;
if (palm_init(&opt, &h) != PALM_OK) { puts(palm_last_error(NULL)); return; }
PalmFrame f = {rgba, rgba_bytes, width, height, row_stride, PALM_FMT_RGBA, rotation_degrees, 0};
char id[PALM_ID_MAX];
float score;
int s = palm_check_register(h, &f, id, sizeof id, &score);
if (s == PALM_OK) printf("xin chào %s\n", id); else printf("%s\n", palm_status_text(s));
palm_release(h);
```

Quy ước:
- Mọi hàm trả mã `PalmStatus` (0 là OK) và không ném exception. Con trỏ bắt buộc mà NULL thì trả `PALM_ERR_ARGUMENT`; output tùy chọn (`score`, `same`, `frame_status`) được phép NULL. `palm_release(NULL)` không làm gì.
- Buffer `id` phải có ít nhất `PALM_ID_MAX` (64) byte và truyền độ dài vào. ID không bao giờ bị cắt ngắn. Khi lỗi, `id` là chuỗi rỗng, trừ `PALM_ERR_ALREADY_REGISTERED` (khi đó `id` là ID đã đăng ký).
- `PalmFrame`: `size` là số byte đọc được và được kiểm tra với `width`/`height`/`stride`/`format` để không đọc vượt buffer. Định dạng hỗ trợ: BGR, RGB, BGRA, RGBA, NV21 (cạnh phải chẵn). `rotation` là số độ xoay theo chiều kim đồng hồ để ảnh thẳng đứng, đúng nghĩa `ImageInfo.getRotationDegrees()` của Android. `mirrored = 1` chỉ dùng cho ảnh đã lật gương (ảnh xem trước selfie); buffer camera, kể cả camera trước, đều không lật.
- `palm_register` nhận 1..32 khung hình, nên dùng 5 khung chụp cách nhau khoảng 0.7 s và nghiêng tay giữa các lần chụp. Cần ít nhất `template.min_valid` khung hợp lệ và nhất quán. Nếu bàn tay khớp một người đã đăng ký (trên `dup_cos_max`) thì trả `PALM_ERR_ALREADY_REGISTERED`. `frame_status` (NULL hoặc mảng ≥ n phần tử) cho biết lý do từng khung bị loại.
- `palm_check_register` chỉ xét 1 khung hình; khoảng 16% khung đơn của người đã đăng ký bị từ chối ở ngưỡng 0.82. Khi đăng nhập, nên thử vài khung liên tiếp và yêu cầu vài lần OK cùng một ID. Khi không nhận ra, hàm không tiết lộ ID gần nhất.
- Mỗi handle có mutex riêng nên gọi từ nhiều thread vẫn an toàn (các lệnh chạy tuần tự). Chuỗi từ `palm_last_error(h)` chỉ hợp lệ đến lần gọi kế tiếp trên `h`.
- Gallery được nạp lại khi file thay đổi (`palm_gui`/`palm_cli` cùng ghi). Nếu file không đọc được: check tiếp tục dùng bản trong RAM, còn register trả `PALM_ERR_IO` để không ghi đè. Mỗi lần ghi là một transaction SQLite (process khác đang giữ khóa quá 5 giây thì báo lỗi `PALM_ERR_IO`), nhưng mỗi lần lưu thay cả nội dung gallery: hai process cùng đăng ký một lúc thì bên ghi sau thắng.

Mã lỗi: `ARGUMENT` 1, `INIT` 2, `IO` 3, `INTERNAL` 4; khung hình không dùng được: `NO_HAND` 10, `LEFT_HAND` 11, `BACK_OF_HAND` 12, `HAND_TOO_SMALL` 13, `HAND_OUT_OF_FRAME` 14, `BLURRY` 15; register: `TOO_FEW_SAMPLES` 20, `ALREADY_REGISTERED` 21; check: `GALLERY_EMPTY` 30, `NOT_RECOGNIZED` 31. `palm_status_text()` trả mô tả tiếng Anh.

ID tự sinh lớn hơn mọi `user_<số>` từng có trong gallery, kể cả người đã xóa, nên không bao giờ cấp lại ID cũ; bộ đếm được lưu trong database (bảng `meta`). Nếu lưu gallery lỗi (`PALM_ERR_IO`), số ID vừa sinh bị bỏ trống, giống auto-increment của CSDL.

Build `.so` cho Android (chưa làm): thêm `add_library(palm_android SHARED src/palm_api.cpp ...)` build bằng toolchain NDK, cùng OpenCV Android SDK và ONNX Runtime AAR, rồi viết wrapper JNI gọi các hàm trên. Model phải được chép từ assets ra thư mục file của app để có đường dẫn.

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
- Các số trên đo với template trung bình của 5 mẫu đầu và chưa giới hạn góc nghiêng (trước 2026-10-09). Xem mục "Góc nghiêng lòng bàn tay" bên dưới.
- Có ROI hỏng: người 22 bị mờ, người 24 trông như da mặt (xem `eval/rois.jpg`). `palm_gui` loại ảnh mờ (`blur_min`), ảnh rung và bàn tay nằm ngoài khung.

## Góc nghiêng lòng bàn tay

Đặc trưng EE-PRnet của cùng một bàn tay thay đổi theo góc nghiêng **ra khỏi mặt phẳng ảnh**. Xoay trong mặt phẳng ảnh thì gần như không ảnh hưởng (xoay 30° vẫn cho cos 0.96). Đo trên `output/original` (324 ảnh tay phải, 24 người), góc nghiêng lấy từ pháp tuyến lòng bàn tay của world landmarks MediaPipe, nhiễu đo khoảng 2–5°:

| Lệch góc giữa 2 ảnh | cos cùng người (trung vị) | tỷ lệ < 0.82 |
|---|---|---|
| 0–5° | 0.976 | 4% |
| 5–10° | 0.915 | 21% |
| 10–15° | 0.855 | 29% |
| 15–20° | 0.751 | 79% |
| 20–30° | 0.736 | 99% |

Người khác nhau: trung vị 0.535, p99 0.746. Hệ quả là cùng một người đăng ký ở các góc khác nhau thì không bị coi là trùng: đã xảy ra với user_0008..0014 ngày 2026-10-09, cos giữa các template chỉ 0.29–0.81.

Cách xử lý hiện tại:
- Cổng chụp đo góc nghiêng (`capture.max_tilt_deg`, 25°) và áp dụng cho cả đăng ký lẫn nhận dạng. Trên tập trên, giới hạn này đưa EER từ 9.8% xuống 2.3–2.9%.
- Đăng ký: mẫu đầu phải gần thẳng (`frontal_max_deg`). Mỗi mẫu sau phải lệch ít nhất `min_pose_sep_deg` so với mọi mẫu đã chụp. `palm_gui` hiện một bản đồ góc ở góc phải trên để hướng dẫn.
- Gallery lưu **từng mẫu** (bảng `samples`) thay vì vector trung bình. Nhận dạng lấy cos lớn nhất với các mẫu của mỗi người; kiểm tra trùng lấy cos lớn nhất giữa mọi cặp (mẫu mới, mẫu đã lưu).
- Khoảng sáng được thu hẹp còn 70..160.

Ngưỡng 0.82 chưa đo lại cho cách khớp theo mẫu trên dữ liệu đa góc, vì chưa có dữ liệu như vậy. `frontal_max_deg`, `min_pose_sep_deg` và khoảng sáng mới chỉ là ước lượng: hiệu chỉnh theo các dòng HAND/ENROLL trong log (`nghiêng`, `lệch`, `sáng`). C API (`palm_register`) đã lưu từng mẫu, nhưng **chưa** kiểm tra góc nghiêng của các khung nhận vào.

## Khác với tài liệu hướng dẫn

- Dùng một ONNX duy nhất thay cho `theta.onnx`, `embed.onnx` và `tps_T.bin`, vì TPS và sampler đã nằm trong graph.
- Không dùng yaml-cpp, Catch2, Eigen hay libsodium (máy chưa cài). Thay vào đó: `cv::FileStorage` cho YAML, test tự viết chạy bằng CTest, `std::vector` cho vector đặc trưng.
- `gallery.db` (SQLite) **chưa mã hóa**. Phải thêm mã hóa (libsodium hoặc SQLCipher) trước khi chạy với dữ liệu thật.
- Gallery là một database SQLite (`gallery.db`, amalgamation trong `third_party/sqlite3`, build kèm `libpalm`): bảng `meta` (`model_version`, `dim`, `next_seq`), `users`, `samples` (mỗi mẫu là một BLOB `dim` float32 little-endian, `PRAGMA user_version` = 1). Định dạng `.bin` (`PALMGAL1/2/3`) đã bỏ hẳn: code không còn đọc hay ghi nó. Gallery 5 người cũ được chuyển sang `.db` một lần (2026-10-13) bằng công cụ tạm không nằm trong repo. Chương trình đã build trước đó không đọc được `.db`, nên cần `cmake --install` lại vào `binary/`.

## Chưa làm

- Mã hóa gallery.
- Liveness.
- Kiểm tra góc nghiêng trong C API (`palm_register`, `palm_check_register`).
- Đo lại `accept_cos_min`/`dup_cos_max` cho cách khớp theo mẫu, trên dữ liệu đăng ký đa góc.
- Đo các ngưỡng `max_motion` và `min_crop_frac` trên camera thật: hiện mới là giá trị khởi đầu.
