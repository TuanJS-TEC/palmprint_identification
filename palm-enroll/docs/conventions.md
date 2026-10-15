# Quy ước đã chốt

Những điều rút ra từ việc đối chiếu với mã MATLAB gốc và từ dữ liệu. Đổi chúng mà không đo lại thì kết quả nhận dạng sẽ xấu đi.

| Mục | Giá trị | Nguồn |
| --- | --- | --- |
| Ảnh vào | Crop cả bàn tay phải, RGB 0–255; trừ mean ImageNet nằm trong ONNX | `buildFeat.m` |
| Resize 56×56 và 224×224 | Như `vl_imreadjpeg`: crop giữa theo tỉ lệ khung đầu ra, lọc tam giác float nở theo tỉ lệ thu nhỏ, biên replicate. **Không** dùng `cv::resize` | `vl_imreadjpeg.cu`, `imread_helpers.hpp` |
| `input_224` | Luôn 224×224 như lúc train, kể cả checkpoint ghi `imgOrg` | `getBatchNclassSpecial.m` |
| θ (18 số) | `[x1..x9, y1..y9]`, tọa độ tuyệt đối trong [-1, 1], 9 điểm trên lưới 3×3 thứ tự cột của MATLAB | `TpsGridGenerator.m` |
| Lấy mẫu TPS | `align_corners=True`, ngoài ảnh = 0 sau khi trừ mean; nằm trong ONNX, C++ không cần `tps_T.bin` | `vl_nnbilinearsampler` |
| Tay phải trên camera thật | Khung camera không lật: nhãn detector **"Right"** và tích chéo (ngón trỏ, ngón út quanh cổ tay) < 0. Nhãn và hình học khớp nhau ở 97.6% ảnh | `palm_handcheck` |
| Hướng ảnh | Lòng bàn tay phải nhìn thẳng, ngón cái bên phải ảnh, như NTU-PI-v1. Không lật ảnh ở đâu cả: khung camera thật không lật, và cổng chụp chỉ cho qua tay phải có lòng bàn tay hướng camera | `palm_eval` |
| Khung cắt bàn tay | Hình vuông tâm trùng tâm khung bao 21 landmark, cạnh 1.17 × max(rộng, cao). Hiệu chỉnh trên 596 crop (p10–p90: 1.14–1.20) | so khớp `hand_crop` với ảnh gốc |

# Khác với tài liệu hướng dẫn ban đầu

- Dùng một ONNX duy nhất thay cho `theta.onnx`, `embed.onnx` và `tps_T.bin`, vì TPS và sampler đã nằm trong graph.
- Không dùng yaml-cpp, Catch2, Eigen hay libsodium (máy chưa cài). Thay vào đó: `cv::FileStorage` cho YAML, `std::vector` cho vector đặc trưng.
- `gallery.db` (SQLite) **chưa mã hóa**. Phải thêm mã hóa (libsodium hoặc SQLCipher) trước khi chạy với dữ liệu thật.
- Gallery là một database SQLite (`gallery.db`, amalgamation trong `third_party/sqlite3`, build kèm thư viện `palm`): bảng `meta` (`model_version`, `dim`, `next_seq`), `users`, `samples` (mỗi mẫu là một BLOB `dim` float32 little-endian, `PRAGMA user_version` = 1). Định dạng `.bin` (`PALMGAL1/2/3`) đã bỏ hẳn: code không còn đọc hay ghi nó. Gallery 5 người cũ được chuyển sang `.db` một lần (2026-10-13) bằng công cụ tạm không nằm trong repo. Chương trình build trước đó không đọc được `.db`, nên cần `cmake --install` lại vào `binary/`.
