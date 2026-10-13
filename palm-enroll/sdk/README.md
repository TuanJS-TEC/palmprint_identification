# Palm SDK: đăng ký và đăng nhập bằng lòng bàn tay phải

SDK C/C++ nhận **khung hình camera** (RGBA, BGR, NV21…) rồi tự làm các bước: phát hiện bàn tay, kiểm tra đó là lòng bàn tay **phải**, cắt vùng bàn tay, trích vector đặc trưng 512 chiều bằng model EE-PRnet, sau đó so khớp với CSDL (file gallery). Giao diện là C thuần ([include/palm_api.h](include/palm_api.h)), gói lại bằng JNI là dùng được trên Android.

> Đọc [NOTICE.md](NOTICE.md) trước: model nhận dạng chưa có giấy phép dùng thương mại, và ảnh trong `testdata/` là dữ liệu cá nhân.

## Nội dung gói

```
include/palm_api.h      C API: toàn bộ hợp đồng sử dụng nằm trong file này
src/                    mã nguồn C++17 (phụ thuộc: OpenCV core + imgproc, ONNX Runtime)
CMakeLists.txt, palm.map  build libpalm.so (chỉ xuất 8 hàm palm_*) và libpalm_static.a
models/                 3 model ONNX (~148 MB) + SHA256SUMS + giấy phép MediaPipe
config.yaml             ngưỡng đã đo (tùy chọn: không truyền config thì dùng đúng các giá trị này)
tests/test_golden.cpp   kiểm tra bản build cho ra đúng kết quả như máy tham chiếu
testdata/               8 ảnh + kết quả chuẩn (expected.txt, features/*.f32)
tools/palm_cli.cpp      công cụ dòng lệnh trên desktop (register / check / list / remove)
```

## API

| Hàm | Việc |
| --- | --- |
| `palm_init(&opt, &h)` / `palm_release(h)` | Nạp 3 model và gallery. Truyền đường dẫn 3 file `.onnx` và `gallery_path` (hoặc `config_path`) |
| `palm_extract(h, &frame, feat, 512)` | Khung hình → vector 512 chiều đã chuẩn hóa |
| `palm_compare(h, a, b, 512, &score, &same)` | Cosine của 2 vector; `same = score >= 0.82` |
| `palm_register(h, frames, n, id, 64, &score, status)` | **Đăng ký**: 1–32 khung của người mới → ID mới `user_0001`, `user_0002`… Nếu bàn tay đã có trong CSDL: `PALM_ERR_ALREADY_REGISTERED` kèm ID cũ |
| `palm_check_register(h, &frame, id, 64, &score)` | **Đăng nhập**: 1 khung → ID, hoặc `PALM_ERR_NOT_RECOGNIZED` |
| `palm_status_text(s)`, `palm_last_error(h)` | Mô tả mã lỗi (tiếng Anh) và chi tiết lỗi của lần gọi trước |

Quy ước (chi tiết trong header):
- Mọi hàm trả mã trạng thái (0 là OK) và không ném exception.
- Con trỏ bắt buộc mà NULL thì trả `PALM_ERR_ARGUMENT`.
- Buffer `id` phải ≥ `PALM_ID_MAX` (64) byte và phải truyền độ dài vào.
- `PalmFrame.size` là số byte đọc được trong buffer, dùng để chặn đọc vượt buffer.
- SDK không bao giờ ghi vào buffer khung hình của app.
- Mỗi handle có mutex riêng, nên gọi từ nhiều thread vẫn an toàn (các lệnh chạy tuần tự). Dù vậy nên dồn mọi lệnh về một thread nền.

Mã trạng thái và gợi ý câu hiển thị cho người dùng:

| Mã | Tên | Câu gợi ý |
| --- | --- | --- |
| 10 | `PALM_ERR_NO_HAND` | Đưa tay phải vào khung hình |
| 11 | `PALM_ERR_LEFT_HAND` | Đây là tay TRÁI, hãy dùng tay PHẢI |
| 12 | `PALM_ERR_BACK_OF_HAND` | Quay LÒNG bàn tay phải về phía camera |
| 13 | `PALM_ERR_HAND_TOO_SMALL` | Đưa tay lại gần camera hơn |
| 14 | `PALM_ERR_HAND_OUT_OF_FRAME` | Đưa cả bàn tay vào giữa khung hình |
| 15 | `PALM_ERR_BLURRY` | Ảnh bị mờ, giữ yên tay |
| 20 | `PALM_ERR_TOO_FEW_SAMPLES` | Chưa đủ ảnh tốt, thử đăng ký lại |
| 21 | `PALM_ERR_ALREADY_REGISTERED` | Bàn tay này đã được đăng ký (ID trong `id`) |
| 30 | `PALM_ERR_GALLERY_EMPTY` | Chưa có ai đăng ký |
| 31 | `PALM_ERR_NOT_RECOGNIZED` | Không nhận ra |
| 1–4 | `ARGUMENT`, `INIT`, `IO`, `INTERNAL` | Lỗi lập trình hoặc hệ thống; xem `palm_last_error(h)` |

## 1. Build và kiểm tra trên Linux (nên làm trước)

Cần CMake ≥ 3.18, compiler C++17, OpenCV 4 (core, imgproc, imgcodecs) và ONNX Runtime 1.16.x.
```bash
cmake -S . -B build -DORT_ROOT=/đường/dẫn/onnxruntime-linux-x64-1.16.3
cmake --build build -j
./build/test_golden models testdata            # phải in PASSED
./build/palm_cli register --gallery /tmp/g.bin testdata/a_1.png testdata/a_2.png testdata/a_3.png testdata/a_4.png testdata/a_5.png
./build/palm_cli check    --gallery /tmp/g.bin testdata/a_probe.png testdata/stranger.png
```

## 2. Build cho Android (NDK)

Gói **chưa được build thử bằng NDK**. Lệnh dưới đây là điểm xuất phát.

Chuẩn bị:
- Android NDK r26 trở lên.
- OpenCV Android SDK 4.x (`OpenCV-android-sdk/sdk/native/jni`).
- `onnxruntime-android-<ver>.aar` từ Maven Central (`com.microsoft.onnxruntime:onnxruntime-android`), giải nén ra (bên trong có `headers/` và `jni/<abi>/libonnxruntime.so`).

```bash
cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DANDROID_STL=c++_shared \
  -DOpenCV_DIR=$OPENCV_ANDROID/sdk/native/jni \
  -DORT_ROOT=$HOME/onnxruntime-android-1.16.3 \
  -DPALM_BUILD_TOOLS=ON          # thêm test_golden để chạy qua adb; tắt khi build cho app
cmake --build build-android
```

Tích hợp vào app bằng Gradle `externalNativeBuild`:
- Trong CMake của app thêm `add_subdirectory(<palm-sdk>)`, viết `palm_jni.cpp`, rồi liên kết với `palm` (giữ hai file `.so` riêng) hoặc với `palm_static` (gộp thành một `.so`).
- Thêm `implementation("com.microsoft.onnxruntime:onnxruntime-android:<cùng phiên bản>")`. Gói này đưa `libonnxruntime.so` vào APK; phiên bản phải khớp với header đã dùng khi build.

**Chạy golden test trên điện thoại.** Bước này cho biết bản port có ra đúng kết quả không, trước khi viết UI:
```bash
adb shell mkdir -p /data/local/tmp/palm
adb push build-android/test_golden build-android/libpalm.so $ORT/jni/arm64-v8a/libonnxruntime.so \
         $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
         /data/local/tmp/palm/
adb push models testdata /data/local/tmp/palm/
adb shell "cd /data/local/tmp/palm && LD_LIBRARY_PATH=. ./test_golden models testdata"
```
Kết quả đúng là `PASSED`:
- vector đặc trưng phải có cos ≥ 0.999 so với `features/*.f32`;
- mã trạng thái và ID phải trùng;
- điểm số chỉ được lệch tối đa 0.01.

Dòng `palm_extract: … ms` cho biết tốc độ trên máy đó.

## 3. Dùng trong app

**Khởi tạo.** Chép 3 file `.onnx` từ `assets/` ra `filesDir` ở lần chạy đầu; nên khai báo `noCompress += "onnx"` để chép nhanh. Sau đó gọi `palm_init` trên thread nền (mất khoảng 1–2 s), với `gallery_path = filesDir/gallery.db`. Giữ một handle duy nhất cho cả app.

**Camera.** Dùng CameraX `ImageAnalysis` với:
- định dạng `OUTPUT_IMAGE_FORMAT_RGBA_8888`, độ phân giải khoảng 1280×720, `STRATEGY_KEEP_ONLY_LATEST`;
- truyền vào `PalmFrame`:
  - `data` = `planes[0].buffer` (buffer trực tiếp, không cần copy);
  - `stride` = `planes[0].rowStride`;
  - `rotation` = `imageInfo.rotationDegrees`;
  - **`mirrored = 0`**: buffer của camera trước không bị lật, chỉ ảnh xem trước bị lật.

Có thể dùng NV21 nếu đã có sẵn; chiều rộng và chiều cao phải chẵn.

**Đăng ký.**
1. Với mỗi khung, gọi `palm_extract` để hiện lời nhắc theo mã trả về.
2. Khi trả OK và đã cách lần giữ trước ≥ 0.7 s, **chép** buffer lại (phải chép trước `image.close()`). Nhắc người dùng nghiêng nhẹ tay giữa các lần. API chưa kiểm tra góc nghiêng, nên app phải tự đảm bảo các khung khác góc nhau (xem README chính, mục "Góc nghiêng lòng bàn tay").
3. Đủ 5 khung thì gọi `palm_register`. Mảng `frame_status` cho biết khung nào bị loại và vì sao.

**Đăng nhập.** Gọi `palm_check_register` trên từng khung. Chỉ coi là thành công khi **3 khung liên tiếp** cùng trả OK với cùng một ID, vì khoảng 16% khung đơn của người đã đăng ký bị từ chối nhầm. Khi không nhận ra, hàm không tiết lộ ID gần nhất.

Ví dụ JNI và Kotlin (minh họa, chưa biên dịch):
```cpp
extern "C" JNIEXPORT jint JNICALL
Java_vn_dms_palm_PalmNative_checkRegister(JNIEnv* env, jclass, jlong h, jobject buf, jint w, jint ht,
                                          jint rowStride, jint rotation, jobjectArray idOut, jfloatArray scoreOut) {
  PalmFrame f{};
  f.data = static_cast<const unsigned char*>(env->GetDirectBufferAddress(buf));
  f.size = static_cast<size_t>(env->GetDirectBufferCapacity(buf));
  f.width = w; f.height = ht; f.stride = rowStride;
  f.format = PALM_FMT_RGBA; f.rotation = rotation; f.mirrored = 0;
  char id[PALM_ID_MAX];
  float score = 0;
  const int s = palm_check_register(reinterpret_cast<PalmHandle>(h), &f, id, sizeof id, &score);
  env->SetObjectArrayElement(idOut, 0, env->NewStringUTF(id));
  env->SetFloatArrayRegion(scoreOut, 0, 1, &score);
  return s;
}
```
```kotlin
analysis.setAnalyzer(palmExecutor) { image ->
    val p = image.planes[0]
    val id = arrayOfNulls<String>(1); val score = FloatArray(1)
    val s = PalmNative.checkRegister(handle, p.buffer, image.width, image.height,
                                     p.rowStride, image.imageInfo.rotationDegrees, id, score)
    image.close()
    // s == 0 → id[0]; ngược lại hiện câu gợi ý theo mã s
}
```

**Kiểm tra nhanh trong app, không cần adb.** Đọc `testdata/a_1.png` bằng `BitmapFactory` (ARGB_8888, lưu trong bộ nhớ theo thứ tự byte RGBA), chép ra direct ByteBuffer, gọi `palm_extract`, rồi `palm_compare` với `features/a_1.f32` (512 float little-endian). Kết quả đúng là cos ≥ 0.999.

## Độ chính xác, tốc độ và những gì cần đo lại

**Độ chính xác.** Đo trên camera USB, 25 người, tay phải:
- xếp đúng người ở vị trí đầu 88.8%, EER 9.8%;
- ở ngưỡng 0.82: 84% khung đơn được chấp nhận đúng và 0 trường hợp chấp nhận nhầm.

Tập đo nhỏ, nên số thật có thể kém hơn.

**Tốc độ.** Trên i5-10210U (4 luồng), mỗi khung mất khoảng 22 ms phát hiện tay và 30 ms trích đặc trưng. Điện thoại tầm trung dự kiến chậm hơn vài lần; luôn chạy ngoài UI thread.

**Ngưỡng cần đo lại trên điện thoại.** Các ngưỡng trong `config.yaml` được đo trên webcam:
- `accept_cos_min` 0.82;
- `blur_min` 100;
- `min_crop_frac` 0.33: cạnh khung cắt bàn tay phải ≥ 0.33 × chiều cao khung đã xoay. Khi cầm máy dọc, chiều cao là cạnh dài, nên tay phải đưa khá gần camera.

Muốn đổi ngưỡng thì chép `config.yaml` ra `filesDir`, sửa, rồi truyền vào `config_path`.

**Model 140 MB (fp32).** Làm APK nặng. Hướng xử lý: tải model về ở lần chạy đầu, hoặc chuyển sang fp16/int8 (phải đo lại độ chính xác).

**Trang bộ nhớ 16 KB.** Google Play yêu cầu app nhắm Android 15+ hỗ trợ trang 16 KB. `libpalm.so` đã được build với `max-page-size=16384`. Nếu `libonnxruntime.so` của phiên bản ORT đang dùng chưa đáp ứng thì phải nâng ORT, rồi chạy lại golden test.

**Thiếu so với một hệ đăng nhập thật:**
- chống giả mạo (liveness): ảnh chụp lòng bàn tay có thể qua được;
- mã hóa gallery: hiện gallery là file SQLite thường, nên mã hóa (SQLCipher hoặc khóa từ Android Keystore);
- API xóa người dùng và liệt kê danh sách.

## Gallery (CSDL)

Database SQLite (SQLite 3.54 build kèm trong `libpalm`, không cần thư viện hệ thống), mỗi người lưu các mẫu đăng ký (mỗi mẫu là một vector 512 chiều). Điểm của một người là cos lớn nhất giữa vector cần so và các mẫu của người đó.
- ID tự sinh tăng dần và không bao giờ dùng lại, kể cả của người đã xóa.
- Mỗi lần lưu là một transaction SQLite (journal), nên không bao giờ bị đọc dở; process khác giữ khóa quá 5 giây thì trả `PALM_ERR_IO`.
- Gallery được nạp lại khi file thay đổi.
- Nếu muốn CSDL nằm trên server: app chỉ cần `palm_extract`, gửi vector lên server. Server lưu từng mẫu, so cosine với từng mẫu rồi lấy giá trị lớn nhất: ≥ 0.82 là đăng nhập được, > 0.82 (giữa bất kỳ cặp mẫu mới và mẫu đã lưu) là đăng ký trùng.
