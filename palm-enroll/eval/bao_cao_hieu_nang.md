# Báo cáo đo hiệu năng pipeline nhận dạng lòng bàn tay (palm-enroll)

Ngày đo: 2026-10-06. Dữ liệu thô nằm trong [`perf/`](perf/), cách đo lại ở mục 8.

> **Lưu ý quan trọng.** Mọi số liệu đo trên máy dev **x86 Linux (Intel i5-10210U, 4 nhân / 8 luồng, 7.8 GB RAM)**, không phải trên Android. Trên SoC ARM của điện thoại, thời gian sẽ khác (thường chậm hơn với cùng số nhân) và trình cấp phát bộ nhớ của Android (Scudo / jemalloc) giữ và trả bộ nhớ khác glibc. Các con số dưới đây cho biết **cấu trúc chi phí** của thuật toán (bước nào tốn bao nhiêu, bộ nhớ tĩnh và động bao nhiêu, có rò rỉ hay không). Cần chạy lại cùng các công cụ trên thiết bị Android đích để có số cuối cùng.

## Tóm tắt

TODO
