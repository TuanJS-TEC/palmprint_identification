# Kiến trúc

`palm-enroll` là **lõi** (thư viện C++ `palm`) cộng một công cụ dòng lệnh. Ứng dụng khác dùng lõi qua `include/palm/`.

## Bố cục

```
palm-enroll/
├── include/palm/        API công khai; palm/palm.h include tất cả
├── src/
│   ├── common/          vec.cpp                       Vec, l2normalize, dot
│   ├── recognition/     preprocess.*, palm_model.cpp  resize kiểu vl_imreadjpeg + EE-PRnet (ONNX)
│   ├── detection/       hand_detector.cpp             MediaPipe: palm box, 21 landmark, world landmark
│   │                    capture_gate.cpp              khung hình này đã đủ tốt để lấy embedding chưa
│   ├── gallery/         template.cpp                  chọn mẫu nào được lưu (outlier, min_valid)
│   │                    gallery.cpp                    người đã đăng ký, tìm kiếm, ID tự động
│   │                    gallery_sqlite.cpp             load / save ra SQLite
│   │                    gallery_internal.h             phần dùng chung của hai file trên (không công khai)
│   └── config/          config.cpp                    config.yaml -> palm::Config
├── apps/palm_cli/       công cụ dòng lệnh (embed, enroll, identify, list, remove)
├── third_party/sqlite3/ SQLite amalgamation
├── docs/                kiến trúc, quy ước đã chốt, số đo
├── config.yaml          cấu hình của lõi: model, ngưỡng, cổng chụp
└── CMakeLists.txt
```

Header nằm cạnh nguồn (`src/recognition/preprocess.h`, `src/gallery/gallery_internal.h`) là nội bộ: ứng dụng dùng thư viện không include và không được cài.

## Phụ thuộc giữa các module

```
common (vec) ◄── gallery (template, gallery, gallery_sqlite)
recognition (preprocess, palm_model)      độc lập, chỉ cần OpenCV + ONNX Runtime
detection (hand_detector ◄── capture_gate) độc lập, chỉ cần OpenCV + ONNX Runtime
config ──► capture_gate, palm_model, template      (chỉ để gom thông số vào một struct)
```

`recognition`, `detection` và `gallery` không biết nhau; chỉ ứng dụng dùng thư viện (hoặc `palm_cli`) ghép chúng. ONNX Runtime và SQLite là phụ thuộc **riêng tư** của thư viện: header công khai chỉ lộ OpenCV (`cv::Mat`, `cv::Rect`).

## Luồng dữ liệu

```
khung camera ─► HandDetector::detect ─► CaptureGate::update ─► GateReport (status, box, normal, ...)
                                                                     │ status Ready / framePasses
                                                                     ▼
                                  crop = frame(report.box) ─► PalmModel::run ─► Embedding.feature
                                                                     │ l2normalize
                              enroll: buildTemplate ─► Gallery::upsert ─► Gallery::save
                              identify: Gallery::search ─► so với accept_cos_min
```

`palm_cli` bỏ qua hai bước đầu: nó nhận ảnh đã là crop bàn tay phải.

## Phần chưa nằm trong lõi

Các quy trình dưới đây chưa có trong lõi, nên mỗi ứng dụng phải tự viết và có thể làm khác nhau:
- Phiên đăng ký: gom `enroll_samples` mẫu ở các góc khác nhau (`CaptureGate::setCollecting` / `addTaken`), hết giờ, rồi `buildTemplate`, kiểm tra trùng người (`dup_cos_max`), `upsert`, `save`.
- Phiên nhận dạng: xác nhận khi cùng một ID đạt `accept_cos_min` trên N khung liên tiếp.
- Chặn ghi đè gallery khi không mở được file (`GalleryError::Kind`).

Đưa các quy trình này vào lõi là bước tiếp theo hợp lý (xem `docs/` khi có thiết kế).

## Gallery trên đĩa

SQLite (`PRAGMA user_version` = 1), mỗi lần `load`/`save` mở một kết nối riêng:
- `meta(key, value)`: `model_version`, `dim`, `next_seq`.
- `users(id)`: theo thứ tự đăng ký (rowid).
- `samples(user_id, idx, embedding)`: mỗi mẫu là BLOB `dim` float32 little-endian.

`save` thay toàn bộ nội dung trong **một transaction**: người đọc thấy bản cũ hoặc bản mới, lỗi thì giữ bản cũ. Hệ quả: hai tiến trình cùng mở một `gallery.db` thì lần `save` sau ghi đè thay đổi của lần trước.
