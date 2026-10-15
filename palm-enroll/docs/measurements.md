# Số đo và hiệu chỉnh

Mọi ngưỡng trong `config.yaml` bắt nguồn từ các số đo dưới đây. Công cụ đo (`palm_eval`, `palm_handcheck`) không còn trong repo, nên số liệu chỉ để tham chiếu, không tái lập được.

## Kết quả đo (2026-10-05)

409 ảnh tay phải trong `output/hand_crop` của 25 người. Mỗi người đăng ký bằng 5 mẫu đầu theo thời gian, các mẫu còn lại làm probe.

| Chỉ số | Giá trị |
| --- | --- |
| Rank-1 (25 người, 284 probe) | 89.1% |
| EER probe–template | 9.2% |
| Ngưỡng 0.82 | 0 chấp nhận nhầm, 16.5% ảnh đơn bị từ chối |
| Suy luận / tiền xử lý (máy dev, 4 luồng) | 29 ms / 6 ms mỗi ảnh |
| C++ so với Python ORT | Trùng bit (sai lệch 0) |

Ngưỡng được chọn trên chính tập này và FAR 0.1% chỉ dựa trên khoảng 6 điểm impostor, nên số thật có thể kém hơn.

Đo đúng đường xử lý trên camera (detector, cổng chụp, rồi embedding) trên ảnh gốc, lật lại cho giống khung camera thật:
- 324/410 ảnh qua được các bước kiểm tra; 81 ảnh bị loại vì bàn tay chạm mép ảnh.
- Kết quả: rank-1 88.8%, EER 9.8%; tại ngưỡng 0.82 có 84.2% chấp nhận đúng, 0 chấp nhận nhầm.
- Như vậy tương đương khi dùng `hand_crop`.

Lưu ý:
- Các số trên đo với template trung bình của 5 mẫu đầu và chưa giới hạn góc nghiêng (trước 2026-10-09). Xem mục "Góc nghiêng lòng bàn tay" bên dưới.
- Có ROI hỏng: người 22 bị mờ, người 24 trông như da mặt. Cổng chụp loại ảnh mờ (`blur_min`), ảnh rung và bàn tay nằm ngoài khung.

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
- Đăng ký: mẫu đầu phải gần thẳng (`frontal_max_deg`). Mỗi mẫu sau phải lệch ít nhất `min_pose_sep_deg` so với mọi mẫu đã chụp. Nên hiển thị góc hiện tại và góc các mẫu đã chụp để hướng dẫn người dùng.
- Gallery lưu **từng mẫu** (bảng `samples`) thay vì vector trung bình. Nhận dạng lấy cos lớn nhất với các mẫu của mỗi người; kiểm tra trùng lấy cos lớn nhất giữa mọi cặp (mẫu mới, mẫu đã lưu).
- Khoảng sáng được thu hẹp còn 70..160.

Ngưỡng 0.82 chưa đo lại cho cách khớp theo mẫu trên dữ liệu đa góc, vì chưa có dữ liệu như vậy. `frontal_max_deg`, `min_pose_sep_deg` và khoảng sáng mới chỉ là ước lượng: hiệu chỉnh theo các số đo `nghiêng`, `lệch`, `sáng` mà `GateReport` trả về cho từng khung.
