#!/bin/sh
# Chạy palm_gui: đăng ký / nhận dạng lòng bàn tay PHẢI bằng camera.
#   ./run_gui.sh          màn hình gắn vào máy (EGLFS, không cần X);
#                         trong phiên X/Wayland thì mở cửa sổ bình thường
#   ./run_gui.sh --vnc    xem và điều khiển từ xa bằng VNC viewer: <ip-máy>:5900
# Các tham số còn lại chuyển cho palm_gui, ví dụ: ./run_gui.sh --source anh.jpg
cd "$(dirname "$0")" || exit 1
if [ "$1" = "--vnc" ]; then
  shift
  export QT_QPA_PLATFORM="vnc:size=1366x768"
  echo "VNC: kết nối tới $(hostname -I 2>/dev/null | cut -d' ' -f1):5900"
elif [ -n "$DISPLAY" ]; then
  export QT_QPA_PLATFORM=xcb
elif [ -n "$WAYLAND_DISPLAY" ]; then
  export QT_QPA_PLATFORM=wayland
else
  export QT_QPA_PLATFORM=eglfs
  export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
  export QT_QPA_EGLFS_KMS_CONFIG="$PWD/eglfs_kms.json"
fi
exec ./palm_gui config.yaml "$@"
