# Nguồn gốc và giấy phép

| Thành phần | Nguồn | Giấy phép |
| --- | --- | --- |
| `src/`, `include/`, `tools/`, `tests/` | Dự án DMS palmprint | Nội bộ |
| `models/palm_detection_mediapipe_2023feb.onnx`, `models/handpose_estimation_mediapipe_2023feb.onnx` | Model bàn tay của MediaPipe (Google), OpenCV Zoo chuyển sang ONNX | Apache-2.0, xem `models/LICENSE.mediapipe-hand-models` |
| `models/EE_PRnet-epoch-60.onnx` | Chuyển từ `EE_PRnet-epoch-60.mat` của Matkowski, Chai, Kong, "Palmprint Recognition in Uncontrolled and Uncooperative Environment", IEEE TIFS 2019, github.com/matkowski-voy/Palmprint-Recognition-in-the-Wild | **Không có giấy phép**, xem bên dưới |
| `testdata/*.png` | Ảnh camera của bộ thu dữ liệu nội bộ | **Dữ liệu cá nhân**, xem bên dưới |
| `third_party/sqlite3/` | SQLite 3.54.0 amalgamation, sqlite.org | Public domain |
| OpenCV, ONNX Runtime (phụ thuộc, không nằm trong gói) | opencv.org, onnxruntime.ai | Apache-2.0, MIT |

## Phải đọc trước khi đưa vào sản phẩm

**EE-PRnet (model nhận dạng).**
- **Không có giấy phép:** repo của tác giả không kèm giấy phép nào, nên mặc định không có quyền phân phối lại weights.
- **Dữ liệu huấn luyện cấm thương mại:** model được huấn luyện trên bộ NTU Palmprints from the Internet (NTU-PI-v1). Thỏa thuận sử dụng của bộ này:
  - cấm mọi mục đích thương mại (Annex A, điều 3);
  - chỉ cho lưu "kết quả nghiên cứu suy ra từ bộ dữ liệu" trên hạ tầng CNTT của chính tổ chức, không lưu trên iCloud, Google Drive, Dropbox… (điều 7).

Hệ quả:
- Gói này chỉ dùng để phát triển và thử nghiệm nội bộ. Gửi và lưu gói qua hạ tầng nội bộ, không qua Google Drive, Dropbox hay dịch vụ lưu trữ bên ngoài.
- Muốn phát hành app có model này cho người dùng thì trước đó phải có xác nhận bằng văn bản của tác giả hoặc NTU (forensics@ntu.edu.sg), hoặc thay bằng model tự huấn luyện trên dữ liệu được phép dùng.

**Ảnh mẫu trong `testdata/`.** Đây là ảnh thật của người trong bộ thu nội bộ, có lòng bàn tay và khuôn mặt, tức là dữ liệu sinh trắc học cá nhân. Chỉ dùng để kiểm tra SDK. Không đưa vào app, kho mã công khai hay tài liệu, và xóa khi không còn cần.
