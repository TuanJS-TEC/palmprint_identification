PALMPRINT GUI - đăng ký / nhận dạng lòng bàn tay PHẢI bằng camera
==================================================================

CHẠY
  ./run_gui.sh           hiện lên màn hình gắn vào máy (chạy được cả từ SSH, không cần X)
  ./run_gui.sh --vnc     xem và điều khiển từ xa: mở VNC viewer tới <ip-máy>:5900
  Thoát: phím Esc hoặc nút "Thoát". Mỗi lần chỉ chạy được 1 app dùng camera.

ĐĂNG KÝ (mỗi người 1 lần, chỉ tay PHẢI)
  1. Nhập ID (mặc định user_01, user_02, ...) rồi bấm "Bắt đầu đăng ký".
  2. Giơ tay PHẢI, lòng bàn tay hướng vào camera, xòe các ngón, cả bàn tay nằm
     trong hình. Khung quanh tay màu xanh = đạt; màu cam/đỏ = làm theo dòng chữ
     phía trên (đưa lại gần, giữ yên, quay lòng bàn tay, dùng tay phải, ...).
  3. App tự chụp 5 mẫu, cách nhau 0.7 giây. Giữa các lần nghiêng/xoay nhẹ tay.
  4. Đăng ký lại cùng ID: app hỏi có thay mẫu cũ không. Cùng một bàn tay dưới
     ID khác: bị từ chối.

NHẬN DẠNG
  Bấm "Bắt đầu nhận dạng" rồi giơ tay phải.
  Xanh "Xin chào <ID>": 3 khung hình liên tiếp đạt cos >= 0.82 với cùng một ID.
  Vàng "<ID> ?": đang xác nhận.  Đỏ "Không nhận ra": dưới ngưỡng.
  Hai ô ảnh: bàn tay đưa vào model và vùng lòng bàn tay model tự căn chỉnh.

TỆP
  palm_gui, palm_cli   chương trình; palm_cli quản lý gallery bằng dòng lệnh
                       (./palm_cli list, ./palm_cli remove --id user_01)
  config.yaml          camera, ngưỡng; ngưỡng đo trên dữ liệu output/ của dự án
  gallery.bin          người đã đăng ký, tạo ở lần đăng ký đầu (CHƯA mã hóa)
  models/              EE_PRnet (nhận dạng) + 2 model MediaPipe phát hiện tay (Apache-2.0)

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
