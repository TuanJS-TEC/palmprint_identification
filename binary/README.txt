PALMPRINT GUI - đăng ký / nhận dạng lòng bàn tay PHẢI bằng camera
==================================================================

CHẠY
  ./run_gui.sh           hiện lên màn hình gắn vào máy (chạy được cả từ SSH, không cần X)
  ./run_gui.sh --vnc     xem và điều khiển từ xa: mở VNC viewer tới <ip-máy>:5900
  Thoát: phím Esc hoặc nút "Thoát". Mỗi lần chỉ chạy được 1 app dùng camera.

ĐĂNG KÝ (mỗi người 1 lần, chỉ tay PHẢI)
  1. Nhập ID (mặc định user_0001, user_0002, ...) rồi bấm "Bắt đầu đăng ký".
  2. Đưa tay PHẢI vào khung đặt tay ở giữa hình: lòng bàn tay hướng vào camera,
     xòe các ngón tự nhiên, cả bàn tay nằm trong hình. Viền khung: đỏ = chưa thấy
     tay phải; cam = làm theo dòng chữ phía trên (đưa tay gần hơn, lùi tay ra,
     xoay lòng bàn tay thẳng hơn, mở ngón tay, đổi vị trí nếu ngược sáng, ...).
  3. Khi đủ điều kiện, viền chuyển xanh và vòng tròn quanh khung chạy dần; giữ yên
     tay cho đến khi vòng đầy thì app tự chụp (màn hình nháy một lần). Chụp 5 mẫu,
     cách nhau 0.7 giây; giữa các mẫu nghiêng/xoay nhẹ tay.
  4. Đăng ký lại cùng ID: app hỏi có thay mẫu cũ không. Cùng một bàn tay dưới
     ID khác: bị từ chối.

NHẬN DẠNG
  Bấm "Bắt đầu nhận dạng" rồi giơ tay phải.
  Xanh "Xin chào <ID>": 3 khung hình liên tiếp đạt cos >= 0.82 với cùng một ID.
  Vàng "<ID> ?": đang xác nhận.  Đỏ "Không nhận ra": dưới ngưỡng.
  Hai ô ảnh: bàn tay đưa vào model và vùng lòng bàn tay model tự căn chỉnh.

TỆP
  palm_gui, palm_cli   chương trình; palm_cli quản lý gallery bằng dòng lệnh
                       (./palm_cli list, ./palm_cli remove --id user_0001)
  config.yaml          camera, ngưỡng; ngưỡng đo trên dữ liệu output/ của dự án
  gallery.bin          người đã đăng ký, tạo ở lần đăng ký đầu (CHƯA mã hóa)
  palm_gui.log         log sự kiện và độ tin cậy (xem mục LOG)
  models/              EE_PRnet (nhận dạng) + 2 model MediaPipe phát hiện tay (Apache-2.0)

LOG
  Khung dưới video ghi từng sự kiện; phím F2 hoặc nút "Log" để ẩn/hiện.
  Cùng nội dung được ghi vào palm_gui.log (cạnh config.yaml, khi quá 10 MB
  đổi tên thành palm_gui.log.1). Xem từ SSH: tail -f palm_gui.log
  HAND    trạng thái tay khi đổi + độ tin cậy của 2 model MediaPipe:
          palm = điểm phát hiện lòng bàn tay, pres = có bàn tay (>= 0.8 mới
          tính), right = xác suất tay PHẢI (> 0.5), box = cạnh khung cắt (px),
          sharp = độ nét (>= blur_min)
  ENROLL  mỗi mẫu: cos với mẫu 1 / mẫu trước, người khác gần nhất; kết quả
  ID      mỗi lần nhận dạng: ID gần nhất + cos, người thứ 2 + khoảng cách,
          ĐẠT k/3 hoặc DƯỚI NGƯỠNG; dòng xanh XÁC NHẬN khi đủ 3 khung liên tiếp
  PERF    mỗi 30 s: fps, thời gian phát hiện tay và embedding
  Model nhận dạng (EE-PRnet) không cho xác suất: độ tin cậy của nó là cos
  (1 = giống hệt). Đo trên ảnh camera: đúng người trung vị 0.95, người khác
  trung vị 0.55 và 99% dưới 0.76; ngưỡng chấp nhận 0.82.
  Tham số: --log FILE để ghi chỗ khác, --log "" để chỉ hiện trên màn hình.

THỬ KHÔNG CẦN CAMERA
  ./run_gui.sh --source anh.jpg             ảnh chụp thẳng từ camera
  ./run_gui.sh --source anh.jpg --mirrored  ảnh đã lật gương (như output/original)

LỖI THƯỜNG GẶP
  "Không mở được camera"  kiểm tra /dev/video0, user thuộc nhóm video, không có
                          chương trình khác đang mở camera.
  Lỗi EGLFS / màn hình đen  đang có X hoặc Wayland chiếm màn hình: chạy trong
                          phiên đó (có DISPLAY) hoặc dùng --vnc.

Cần: OpenCV 4.10 và ONNX Runtime 1.16 trong /usr/local/lib, Qt 5.15 (đã có trên máy này).
Mã nguồn và cách build: ../palm-enroll (cmake --install build --prefix ../binary).
