# Hướng dẫn triển khai quy trình đăng ký lòng bàn tay bằng C++

Oct 5, 2026 · @Tuan Hoang Minh

## Tổng quan

Mục tiêu là một quy trình đăng ký lòng bàn tay viết bằng C++, chạy trên i5-10210U với 8 GB RAM, thu nhiều mẫu đa dạng rồi gộp thành một template 512 chiều.

**Đã xác định** (từ việc đọc trực tiếp hai file .mat):

- Model là MatConvNet DagNN kèm lớp tự viết `dagnnExtra.TpsGridGenerator`, nên không dùng được `exportONNXNetwork` của MATLAB.
- AlignNet nhận ảnh 56×56×3 và trả 18 số: tọa độ (x, y) của 9 điểm điều khiển TPS trên lưới 3×3. Nó không phát hiện tay và không cắt ROI.
- EE\_PRnet có hai đầu vào: ảnh 56×56×3 cho nhánh căn chỉnh và ảnh ROI 224×224×3. Ảnh được warp bằng TPS thành 112×112, rồi nhánh nhận dạng cho vector 512 chiều tại lớp `fc1R` (tuyến tính, không ReLU).
- Trường meta của hai file (224×224, mean ImageNet) chép từ VGG-16, không phản ánh tiền xử lý thật.
- Camera là RGB thường: 811 ảnh 1920×1080 và 20 ảnh 1280×720, độ sáng trung bình 121.5 ± 15.1.

**Chưa kiểm chứng** (mỗi mục có giai đoạn xử lý riêng):

- Có trừ mean ảnh khi huấn luyện hay không (giai đoạn 1).
- Quy ước của `TpsGridGenerator` và `BilinearSampler`: chuẩn hóa tọa độ, thứ tự 9 điểm (giai đoạn 1 và 3).
- Cách cắt ROI trong dữ liệu huấn luyện, tức hệ số `k` và độ lệch (giai đoạn 4 và 5).
- Hand landmarker chạy ổn định trong C++ (giai đoạn 4).
- Thời gian chạy: ước tính lý thuyết khoảng 2.9 GMAC mỗi frame, tức 60–150 ms trên máy này. Đây chưa phải số đo (giai đoạn 9).

Số liệu validation trong file khá kém (top-1 error 70.7% trên val, 1.5% trên train). Vì vậy mọi ngưỡng phải tự đo trên dữ liệu của bạn, không lấy từ bài báo.

**Ngoài phạm vi:** model liveness riêng, giao diện đồ họa hoàn chỉnh và xác thực 1:1 lúc vận hành. Hướng dẫn chỉ phủ phần đăng ký, kèm điểm móc để thêm sau.

**Cách đọc:** mỗi giai đoạn có đầu ra và tiêu chí nghiệm thu. Giai đoạn 1–3 là bước chặn: chưa qua thì chưa nên làm phần sau.

## Giai đoạn 0: Chuẩn bị môi trường

Giai đoạn này chốt công cụ, dữ liệu và cấu trúc thư mục để mọi bước sau chạy lặp lại được.

**Giả định cần xác nhận:** bạn chưa cho biết hệ điều hành. Hướng dẫn giả định Windows 10/11 với MSVC (Visual Studio 2022), CMake và vcpkg. Nếu dùng Linux thì thay MSVC bằng GCC, phần còn lại giữ nguyên.

| Việc | Công cụ | Ghi chú |
| --- | --- | --- |
| Camera, xử lý ảnh | OpenCV 4.x | `VideoCapture`, `Laplacian`, `warpAffine`, `remap`, `cv::dnn` |
| Suy luận | ONNX Runtime (CPU) | Thử OpenVINO sau khi có số đo |
| Đại số vector | Eigen 3 | Cosine, medoid, so khớp 1:N |
| Lưu template | SQLite + libsodium | Mã hóa trường embedding trước khi ghi |
| Cấu hình | yaml-cpp | Mọi ngưỡng ở một file |
| Kiểm thử | Catch2 | Chạy trên ảnh tĩnh, không cần camera |
| Chuyển model (một lần) | Python 3.10+, numpy, scipy, h5py, torch, onnx, onnxruntime | Chỉ dùng offline |

**Dữ liệu cần có trước khi viết mã:**

- 831 ảnh gốc trong `output/` cùng hai thư mục `roi/` và `roi_aligned/` hiện có.
- File `data/manifest.csv` với các cột `path, subject, hand, session, mirrored`. Thiếu nó thì không đo được EER ở giai đoạn 5.
- `AlignNet-epoch-25.mat`, `EE_PRnet-epoch-60.mat`, `hand_landmarker.task`.
- Mã nguồn `dagnnExtra.TpsGridGenerator` nếu còn. Nếu có MATLAB và MatConvNet, xuất đầu ra tham chiếu (θ và vector 512 chiều) cho vài ảnh.

**Cấu trúc thư mục:**

```text
palm-enroll/
  CMakeLists.txt
  vcpkg.json
  config.yaml
  data/manifest.csv
  models/                 # ONNX + hand_landmarker.task, không commit
  python/
    convert_mat.py
    check_equivalence.py
  src/
    capture.{h,cpp}        prefilter.{h,cpp}     hand_landmarker.{h,cpp}
    pose.{h,cpp}           roi.{h,cpp}           tps.{h,cpp}
    embedding.{h,cpp}      quality.{h,cpp}       session.{h,cpp}
    template_store.{h,cpp}                       main.cpp
  tools/
    eval_images.cpp        # chạy pipeline trên thư mục ảnh, thống kê
  tests/
```

**Lưu ý camera trên Windows:** việc khóa auto-exposure và auto-focus qua `cap.set(...)` của OpenCV thường không đáng tin. Hãy đo độ sáng trung bình hai frame cách nhau vài giây trước và sau khi đặt để biết lệnh có tác dụng không. Nếu không, cần dùng API riêng của backend camera (MSMF hoặc DirectShow).

**Nghiệm thu:** dự án rỗng build được trên máy của bạn, và một script đọc `manifest.csv` in ra số ảnh theo từng người, tay và phiên.

## Giai đoạn 1: Chuyển model sang ONNX

Chuyển hai file .mat sang ONNX là bước chặn đầu tiên, và phải chứng minh bản mới cho kết quả trùng với bản MATLAB.

**Quyết định thiết kế:** lớp TPS và bilinear sampler không đưa vào ONNX. Ta xuất hai đồ thị riêng, rồi warp ở C++ bằng một ma trận cố định. Ba sản phẩm của giai đoạn này:

- `theta.onnx`: ảnh 56×56×3 → 18 số θ (nhánh căn chỉnh của EE\_PRnet).
- `embed.onnx`: ảnh ROI đã căn chỉnh 112×112×3 → vector 512 chiều tại `fc1R`, chưa chuẩn hóa.
- `tps_T.bin`: ma trận float32 kích thước (112·112)×9, dùng ở C++ để biến θ thành lưới lấy mẫu.

### 1.1 In cấu trúc thật của mạng

Đừng dựa vào mô tả trong báo cáo kiểm tra. In từng lớp để xác nhận trước khi dựng lại.

```python
# python/dump_net.py
import sys, numpy as np, scipy.io as sio
net = sio.loadmat(sys.argv[1], struct_as_record=False, squeeze_me=True)['net']
P = {str(p.name): np.asarray(p.value) for p in np.atleast_1d(net.params)}
for L in np.atleast_1d(net.layers):
    pn = [str(x) for x in np.atleast_1d(L.params)] if np.size(L.params) else []
    print(f'{str(L.name):12s} {str(L.block.type):28s} in={L.inputs} out={L.outputs}',
          [P[n].shape for n in pn])
```

Chạy cho cả hai file và kiểm tra bốn điều:

- Danh sách lớp khớp mô tả (7 conv, 3 pool, LRN, các lớp FC) và có lớp tên `fc1R`.
- Tên biến vào và ra của `dagnnExtra.TpsGridGenerator` và `dagnn.BilinearSampler`.
- Kernel của `fcr1_4` (7×7) và `fc1R` (14×14): chúng lưu dạng conv, nên không phát sinh vấn đề thứ tự flatten.
- Bias của lớp cuối nhánh căn chỉnh. Nếu gần lưới đơn vị (-1, 0, 1) thì nhiều khả năng θ là tọa độ tuyệt đối khởi tạo từ phép đồng nhất. Nếu gần 0 thì θ có thể là độ lệch. Đây chỉ là gợi ý, kết luận ở bước 1.5.

### 1.2 Bộ dựng lại tổng quát

Bộ này đọc đồ thị từ file và chạy từng lớp, nên không phụ thuộc việc báo cáo mô tả đúng hay sai. Nó chỉ hỗ trợ lớp một-vào một-ra và dừng với lỗi rõ ràng khi gặp lớp lạ.

```python
# python/matdag.py
import numpy as np, scipy.io as sio, torch, torch.nn as nn, torch.nn.functional as F

def ints(x): return [int(v) for v in np.atleast_1d(x).ravel()]
def names(x): return [str(v) for v in np.atleast_1d(x).ravel()] if np.size(x) else []

class MatDag(nn.Module):
    def __init__(self, path):
        super().__init__()
        net = sio.loadmat(path, struct_as_record=False, squeeze_me=True)['net']
        P = {str(p.name): np.asarray(p.value, dtype=np.float32) for p in np.atleast_1d(net.params)}
        self.mods, self.layers = nn.ModuleDict(), []
        for L in np.atleast_1d(net.layers):
            b, t, pn = L.block, str(L.block.type), names(L.params)
            r = dict(name=str(L.name), type=t, inp=names(L.inputs), out=names(L.outputs), b=b)
            if t == 'dagnn.Conv':
                size = ints(b.size)                                   # [H, W, in, out]
                st = ints(b.stride); st = st * 2 if len(st) == 1 else st
                pad = ints(b.pad); pad = pad * 4 if len(pad) == 1 else pad
                assert pad[0] == pad[1] and pad[2] == pad[3]
                assert ints(getattr(b, 'dilate', 1)) in ([1], [1, 1])
                m = nn.Conv2d(size[2], size[3], (size[0], size[1]), stride=tuple(st),
                              padding=(pad[0], pad[2]), bias=len(pn) > 1)
                w = P[pn[0]].reshape(size)                            # khôi phục chiều 1 bị squeeze
                m.weight.data = torch.from_numpy(w.transpose(3, 2, 0, 1).copy())
                if len(pn) > 1:
                    m.bias.data = torch.from_numpy(P[pn[1]].reshape(-1).copy())
                self.mods[r['name']] = m
            elif t == 'dagnn.Pooling':
                assert str(b.method) == 'max'
                ps = ints(b.poolSize); ps = ps * 2 if len(ps) == 1 else ps
                st = ints(b.stride); st = st * 2 if len(st) == 1 else st
                pad = ints(b.pad); pad = pad * 4 if len(pad) == 1 else pad
                self.mods[r['name']] = nn.MaxPool2d(tuple(ps), tuple(st), (pad[0], pad[2]))
            self.layers.append(r)

    def _apply(self, r, x):
        t, b = r['type'], r['b']
        if t in ('dagnn.Conv', 'dagnn.Pooling'): return self.mods[r['name']](x)
        if t == 'dagnn.ReLU': return F.leaky_relu(x, float(getattr(b, 'leak', 0)))
        if t == 'dagnn.DropOut': return x                             # bỏ qua khi suy luận
        if t == 'dagnn.LRN':
            N, kappa, alpha, beta = [float(v) for v in np.atleast_1d(b.param)]
            assert N >= x.shape[1], 'cửa sổ LRN nhỏ hơn số kênh: cần cài đúng công thức'
            return x / (kappa + alpha * (x * x).sum(1, keepdim=True)) ** beta   # chuẩn hóa L2 theo kênh
        raise NotImplementedError(t)

    def forward(self, feeds, want):
        by_out = {o: r for r in self.layers for o in r['out']}
        vals = dict(feeds)
        def get(v):
            if v not in vals:
                r = by_out[v]
                vals[v] = self._apply(r, get(r['inp'][0]))
            return vals[v]
        return get(want)

def find_vars(dag):
    g = lambda s: next(r for r in dag.layers if r['type'].endswith(s))
    fc1 = next(r for r in dag.layers if r['name'] == 'fc1R')
    return g('TpsGridGenerator')['inp'][0], g('BilinearSampler')['out'][0], fc1['out'][0]
    # theta_var, aligned_var, embed_var
```

Những chỗ dễ sai khi chuyển:

| Chỗ | Cách xử lý |
| --- | --- |
| Thứ tự trục trọng số | MatConvNet lưu \[H, W, in, out\]; PyTorch cần \[out, in, H, W\], tức `transpose(3, 2, 0, 1)` |
| `squeeze_me=True` làm mất chiều 1 | `reshape` theo `block.size` trước khi hoán vị |
| Lớp FC | Dựng như conv có kernel bằng kích thước không gian (7×7, 14×14), không dùng `Linear` |
| LRN \[1024, 1e-6, 1, 0.5\] | N lớn hơn số kênh nên là chuẩn hóa L2 theo kênh; code có `assert` để bắt trường hợp khác |
| ReLU | Các lớp FC dùng LeakyReLU với hệ số `leak` đọc từ file (báo cáo ghi 0.01) |
| Dropout | Bỏ qua khi suy luận |
| Tiền xử lý | Thang 0–255, thứ tự RGB, có trừ mean hay không: chốt ở giai đoạn 5 bằng EER, hoặc theo mã huấn luyện gốc |

### 1.3 TPS tham chiếu và ma trận T

Bạn chưa có mã nguồn `dagnnExtra.TpsGridGenerator`, nên đoạn dưới cài TPS theo công thức chuẩn. Ba giả định cần kiểm chứng ở bước 1.5, và nếu có mã gốc thì thay bằng mã gốc:

1. Chín điểm điều khiển đích nằm cố định trên lưới đều \[-1, 1\], đánh số theo hàng.
2. θ là tọa độ nguồn (x, y) của chín điểm đó. Cách xếp 18 số (xen kẽ hay x trước y sau) và việc θ là tuyệt đối hay độ lệch chưa rõ.
3. Tọa độ lưới \[-1, 1\] ứng với tâm của pixel đầu và cuối, tương đương `align_corners=True`.

Với điểm đích cố định, ánh xạ TPS là một phép nhân ma trận: lưới nguồn = T × Q, trong đó T chỉ phụ thuộc kích thước ảnh ra và điểm đích, còn Q (9×2) là θ. Vì vậy ta tính T một lần.

```python
# python/tps.py
import torch
CTRL = torch.tensor([[x, y] for y in (-1., 0., 1.) for x in (-1., 0., 1.)])   # 9x2, giả định 1

def U(r2):                                    # r^2 log r^2, bằng 0 tại r = 0
    return torch.where(r2 == 0, torch.zeros_like(r2), r2 * torch.log(r2.clamp_min(1e-12)))

def tps_T(Ho, Wo, ctrl=CTRL):
    n = ctrl.shape[0]
    K = U(((ctrl[:, None] - ctrl[None]) ** 2).sum(-1))
    P = torch.cat([torch.ones(n, 1), ctrl], 1)                 # n x 3
    L = torch.zeros(n + 3, n + 3)
    L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
    Linv = torch.linalg.inv(L)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, Ho), torch.linspace(-1, 1, Wo), indexing='ij')
    g = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1)       # (Ho*Wo) x 2
    Phi = torch.cat([U(((g[:, None] - ctrl[None]) ** 2).sum(-1)), torch.ones(len(g), 1), g], 1)
    return Phi @ Linv[:, :n]                                   # (Ho*Wo) x n
```

Kích thước T là 12544×9 số float32, khoảng 440 KB. Phần nhân với Q chỉ tốn 12544×9×2 phép nhân cộng mỗi frame.

### 1.4 Xuất ONNX và ma trận T

```python
# python/convert_mat.py
import torch
from matdag import MatDag, find_vars
from tps import tps_T

dag = MatDag('EE_PRnet-epoch-60.mat').eval()
theta_var, aligned_var, embed_var = find_vars(dag)

class Sub(torch.nn.Module):
    def __init__(self, inp, out):
        super().__init__(); self.inp, self.out = inp, out
    def forward(self, x):                      # x: thang 0-255, RGB
        # nếu giai đoạn 5 cho thấy phải trừ mean thì trừ ở đây để C++ không phải biết
        return dag({self.inp: x}, self.out)

torch.onnx.export(Sub('ANInput', theta_var), torch.zeros(1, 3, 56, 56), 'models/theta.onnx',
                  input_names=['x'], output_names=['theta'], opset_version=17)
torch.onnx.export(Sub(aligned_var, embed_var), torch.zeros(1, 3, 112, 112), 'models/embed.onnx',
                  input_names=['x'], output_names=['emb'], opset_version=17)
tps_T(112, 112).numpy().astype('float32').tofile('models/tps_T.bin')
```

Ghi chú: `ANInput` là tên biến theo báo cáo, hãy đối chiếu với đầu ra của `dump_net.py`. Bỏ lớp phân loại 2035 lớp, vì chỉ cần vector ở `fc1R`.

### 1.5 Đối chiếu với MATLAB

Bước này chốt các giả định ở 1.2 và 1.3 bằng số đo. Cả hai bên nhận cùng một ảnh đầu vào, nên nó chỉ kiểm tra việc chuyển đổi, không trả lời câu hỏi có trừ mean ảnh hay không (câu đó để giai đoạn 5 quyết định).

**Nếu có MATLAB và MatConvNet**, xuất đầu ra tham chiếu cho 10–20 ảnh ROI 224×224 do bạn chọn:

```matlab
% matlab/export_ref.m  (cần MatConvNet và thư mục chứa dagnnExtra.*)
run('matconvnet/matlab/vl_setupnn'); addpath('path/to/dagnnExtra');
s = load('EE_PRnet-epoch-60.mat');
net = dagnn.DagNN.loadobj(s.net);
net.mode = 'test'; net.conserveMemory = false;
% bỏ lớp loss và softmax nếu chúng đòi nhãn: net.removeLayer('<tên lớp>');
files = dir('ref_in/*.png');
for i = 1:numel(files)
  im = single(imread(fullfile('ref_in', files(i).name)));   % thang 0-255
  an = imresize(im, [56 56]);                                % giả định, xem ghi chú dưới
  net.eval({'ANInput', an, 'Input224', im});
  v = @(name) net.vars(net.getVarIndex(name)).value;
  theta = v('<theta_var>'); aligned = v('<aligned_var>'); emb = v('<embed_var>');
  save(fullfile('ref_out', [files(i).name '.mat']), 'im', 'an', 'theta', 'aligned', 'emb');
end
```

Thay ba tên biến bằng giá trị `find_vars` in ra. Sau đó chạy so sánh phía Python. Script thử bốn cách hiểu θ (tuyệt đối hay độ lệch, xen kẽ hay x trước y sau) và cho biết cách nào khớp ảnh `aligned` của MATLAB:

```python
# python/check_equivalence.py
import glob, numpy as np, scipy.io as sio, torch, torch.nn.functional as F
from matdag import MatDag, find_vars
from tps import tps_T, CTRL

dag = MatDag('EE_PRnet-epoch-60.mat').eval()
theta_var, aligned_var, embed_var = find_vars(dag)
T = tps_T(112, 112)
chw = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).permute(2, 0, 1)[None]
flat = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32).ravel())

with torch.no_grad():
    for f in sorted(glob.glob('ref_out/*.mat')):
        d = sio.loadmat(f)
        th, th_ref = dag({'ANInput': chw(d['an'])}, theta_var).flatten(), flat(d['theta'])
        print(f, 'theta max|diff| =', (th - th_ref).abs().max().item())
        emb = dag({aligned_var: chw(d['aligned'])}, embed_var).flatten()
        print('   emb cosine =', F.cosine_similarity(emb, flat(d['emb']), dim=0).item())
        for off in (False, True):
            kind = 'độ lệch' if off else 'tuyệt đối'
            for nm, Q in (('xen kẽ', th_ref.reshape(9, 2)), ('x rồi y', th_ref.reshape(2, 9).T)):
                grid = (T @ (Q + CTRL if off else Q)).reshape(1, 112, 112, 2)
                ali = F.grid_sample(chw(d['im']), grid, align_corners=True)
                err = (ali - chw(d['aligned'])).abs().max().item()
                print(f'   warp, theta {kind}, {nm}: max|diff| = {err:.3f}')
```

Cách nào cho sai lệch warp gần 0 là quy ước đúng. Nếu không cách nào khớp, nghi ngờ giả định 1 hoặc 3 (điểm đích hoặc `align_corners`). Hãy thử đảo hàng/cột của `CTRL` và đổi `align_corners`, hoặc tìm mã gốc của `TpsGridGenerator`.

Ghi chú về `ANInput`: báo cáo không nói ảnh 56×56 được tạo từ ảnh 224 bằng cách nào. `imresize` của MATLAB (bicubic, có khử răng cưa) khác `cv::resize`. Hãy xác nhận trong mã huấn luyện, rồi đo xem khác biệt nội suy làm θ lệch bao nhiêu.

**Nếu không có MATLAB**, chỉ còn kiểm tra gián tiếp. Chạy PyTorch trên các ROI trong `manifest.csv`, so cosine của cặp cùng người với cặp khác người, và xem bằng mắt vài ảnh warp có giống một lòng bàn tay không méo. Cách này chỉ bắt được lỗi lớn, không chứng minh hai bản trùng nhau.

### Nghiệm thu giai đoạn 1

Các ngưỡng dưới đây là đề xuất của tôi, hãy chỉnh nếu có lý do:

- θ: sai lệch tuyệt đối tối đa nhỏ hơn 1e-3 trên ít nhất 10 ảnh tham chiếu.
- Ảnh warp: sai lệch tối đa dưới 1 mức xám (thang 0–255) so với `aligned` của MATLAB.
- Embedding: cosine giữa PyTorch và MATLAB lớn hơn 0.9999.
- ONNX so với PyTorch: sai lệch tối đa nhỏ hơn 1e-4 cho cả `theta.onnx` và `embed.onnx`.
- Đã ghi lại bốn quy ước (cách xếp θ, tuyệt đối hay độ lệch, `align_corners`, cách tạo `ANInput`) vào `config.yaml` hoặc README.
- Không có MATLAB: chấp nhận tạm khi kiểm tra gián tiếp đạt, nhưng ghi rõ là chưa đối chiếu và quay lại khi có điều kiện.

## Giai đoạn 2: Khung dự án C++

Giai đoạn này dựng khung build và các interface, để mọi module chạy được cả trên ảnh tĩnh lẫn camera. Chưa cần model nào hoạt động, chỉ cần dự án build và test rỗng chạy.

### 2.1 vcpkg và CMake

```json
{
  "name": "palm-enroll",
  "version-string": "0.1.0",
  "dependencies": ["opencv4", "onnxruntime", "eigen3", "yaml-cpp", "sqlite3", "libsodium", "catch2"]
}
```

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(palm_enroll CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(unofficial-sodium CONFIG REQUIRED)
find_package(Catch2 3 CONFIG REQUIRED)

add_library(palm
  src/capture.cpp src/prefilter.cpp src/hand_landmarker.cpp src/pose.cpp src/roi.cpp
  src/tps.cpp src/embedding.cpp src/quality.cpp src/session.cpp src/template_store.cpp)
target_include_directories(palm PUBLIC src)
target_link_libraries(palm PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime Eigen3::Eigen
  yaml-cpp::yaml-cpp unofficial::sqlite3::sqlite3 unofficial-sodium::sodium)

add_executable(eval_images tools/eval_images.cpp)
target_link_libraries(eval_images PRIVATE palm)
add_executable(enroll_cli src/main.cpp)
target_link_libraries(enroll_cli PRIVATE palm)

enable_testing()
file(GLOB TEST_SRC tests/*.cpp)
add_executable(tests ${TEST_SRC})
target_link_libraries(tests PRIVATE palm Catch2::Catch2WithMain)
add_test(NAME tests COMMAND tests)
```

```text
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
ctest --test-dir build -C Release
```

Tên package và target trong `find_package` có thể khác theo phiên bản vcpkg. Khi cài xong, vcpkg in hướng dẫn sử dụng cho từng thư viện, hãy đối chiếu. Port `opencv4` cũng cần bật tính năng `dnn` nếu bạn chọn hướng C ở giai đoạn 4.

### 2.2 Cấu hình tập trung

Mọi ngưỡng nằm trong một file. Các số dưới đây là **giá trị khởi đầu do tôi đoán**, chưa hiệu chỉnh. Mỗi giai đoạn sau nói rõ ngưỡng nào được thay bằng số đo, và hai ngưỡng quan trọng nhất để trống (`null`) cho đến khi bạn có phân bố genuine/impostor.

```yaml
# config.yaml
camera: {width: 1280, height: 720, lock_exposure: true}
prefilter:                 # giai đoạn 5 hiệu chỉnh
  small_size: [160, 120]
  saturated_frac_max: 0.20
  blur_var_min: 20.0
hand:                      # giai đoạn 4
  want_hand: right         # left | right
  min_palm_width_px: 150
  margin_frac: 0.05
  min_open_ratio: 1.2
roi:                       # giai đoạn 5 chọn k và offset
  k: 2.0
  offset: 0.1
  out_size: 224
quality:                   # giai đoạn 6
  blur_var_min: 50.0
  contrast_min: 25.0
  glare_frac_max: 0.02
  theta_deform_max: 0.5
session:                   # giai đoạn 7
  poses: 5
  stable_frames: 3
  min_gap_ms: 500
  pose_window_ms: 4000
  timeout_ms: 30000
template:                  # giai đoạn 8
  min_valid: 3
  outlier_cos_min: null    # lấy từ phân bố genuine
  dup_cos_max: null        # lấy từ phân bố impostor
  model_version: ee_prnet_e60
```

### 2.3 Kiểu dữ liệu và interface

Mọi module nhận `cv::Mat` hoặc kết quả của module trước, để chạy được trên 831 ảnh tĩnh. Máy trạng thái đăng ký không biết gì về camera.

```cpp
// src/types.h
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

struct HandResult {
  bool found = false;
  bool isRight = false;
  float score = 0.f;
  std::array<cv::Point2f, 21> lm{};       // pixel, ảnh gốc
  std::array<cv::Point3f, 21> world{};    // mét
};

struct Verdict {                          // kết quả kiểm tra của một tầng
  bool ok = true;
  std::string reason;                     // mã lỗi ngắn, UI ánh xạ sang thông báo
};

struct Sample {
  Eigen::VectorXf emb;                    // 512 chiều, đã L2-normalize
  float quality = 0.f;
  cv::Vec3f pose{};                       // pose descriptor, dùng để ép đa dạng
  int64_t tsMs = 0;
};
```

```cpp
// src/interfaces.h
#pragma once
#include "types.h"

class IHandLandmarker {                   // thay được cách cài mà không đụng code khác
 public:
  virtual HandResult detect(const cv::Mat& bgr, int64_t tsMs) = 0;
  virtual ~IHandLandmarker() = default;
};

struct PoseCheck { Verdict v; cv::Vec3f desc; };
PoseCheck checkPose(const HandResult&, cv::Size img, bool wantRight);

cv::Mat cropRoi(const cv::Mat& bgr, const HandResult&);            // 224x224 RGB

class EmbeddingModel {                    // bọc ONNX Runtime + ma trận T
 public:
  EmbeddingModel(const std::string& dir); // theta.onnx, embed.onnx, tps_T.bin
  std::array<float, 18> regressTheta(const cv::Mat& roi224);
  cv::Mat warpTps(const cv::Mat& roi224, const std::array<float, 18>& theta);  // 112x112
  Eigen::VectorXf embed(const cv::Mat& aligned112);                // 512, L2-normalize
};

struct QualityResult { Verdict v; float score; };
QualityResult assessRoi(const cv::Mat& aligned112, const std::array<float, 18>& theta);
```

Số luồng ONNX Runtime đặt bằng số nhân vật lý của máy (4 với i5-10210U) khi tạo session, và đo lại ở giai đoạn 9.

**Nghiệm thu:** dự án build ở chế độ Release, `ctest` chạy qua một test rỗng, `config.yaml` được nạp vào struct cấu hình, và `eval_images` đọc được `manifest.csv` rồi in số ảnh.

## Giai đoạn 3: Embedding và TPS trong C++

Giai đoạn này cài `EmbeddingModel` trong C++ và chứng minh nó cho cùng θ, ảnh warp và vector 512 chiều như bản PyTorch trên cùng ảnh ROI. Hãy làm xong giai đoạn này trước khi động tới hand landmarker.

Thêm vào `EmbeddingModel` trong `interfaces.h` một `struct Impl; std::unique_ptr<Impl> p_;` và destructor khai báo sẵn, để header không phải include ONNX Runtime.

### 3.1 Cài đặt

```cpp
// src/embedding.cpp
#include "interfaces.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {
cv::Mat rgb32(const cv::Mat& bgr8) {             // BGR uint8 -> RGB float, thang 0-255
  cv::Mat f; bgr8.convertTo(f, CV_32FC3); cv::cvtColor(f, f, cv::COLOR_BGR2RGB); return f;
}
std::vector<float> toNchw(const cv::Mat& rgb32f) {
  std::vector<cv::Mat> ch(3); cv::split(rgb32f, ch);
  std::vector<float> out(3 * rgb32f.total());
  for (int c = 0; c < 3; ++c)
    std::memcpy(out.data() + c * rgb32f.total(), ch[c].ptr<float>(), rgb32f.total() * sizeof(float));
  return out;
}
std::vector<float> run(Ort::Session& s, std::vector<float>& in, std::vector<int64_t> shape,
                       const char* inName, const char* outName) {
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value x = Ort::Value::CreateTensor<float>(mi, in.data(), in.size(), shape.data(), shape.size());
  auto out = s.Run(Ort::RunOptions{nullptr}, &inName, &x, 1, &outName, 1);
  const float* p = out[0].GetTensorData<float>();
  return {p, p + out[0].GetTensorTypeAndShapeInfo().GetElementCount()};
}
}  // namespace

struct EmbeddingModel::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "palm"};
  Ort::SessionOptions opt;
  std::unique_ptr<Ort::Session> theta, embed;
  cv::Mat T;                                      // 12544 x 9, CV_32F, từ tps_T.bin
  explicit Impl(const std::string& dir) {
    namespace fs = std::filesystem;
    opt.SetIntraOpNumThreads(4);                  // số nhân vật lý, đo lại ở giai đoạn 9
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    theta = std::make_unique<Ort::Session>(env, (fs::path(dir) / "theta.onnx").c_str(), opt);
    embed = std::make_unique<Ort::Session>(env, (fs::path(dir) / "embed.onnx").c_str(), opt);
    T.create(112 * 112, 9, CV_32F);
    std::ifstream f(fs::path(dir) / "tps_T.bin", std::ios::binary);
    f.read(reinterpret_cast<char*>(T.data), T.total() * sizeof(float));
    if (!f) throw std::runtime_error("tps_T.bin sai kích thước");
  }
};

EmbeddingModel::EmbeddingModel(const std::string& dir) : p_(std::make_unique<Impl>(dir)) {}
EmbeddingModel::~EmbeddingModel() = default;

std::array<float, 18> EmbeddingModel::regressTheta(const cv::Mat& roi224) {
  cv::Mat small;
  cv::resize(roi224, small, {56, 56}, 0, 0, cv::INTER_AREA);   // thay theo cách chốt ở bước 1.5
  auto in = toNchw(rgb32(small));
  auto out = run(*p_->theta, in, {1, 3, 56, 56}, "x", "theta");
  std::array<float, 18> th{};
  std::copy_n(out.begin(), 18, th.begin());
  return th;
}

cv::Mat EmbeddingModel::warpTps(const cv::Mat& roi224, const std::array<float, 18>& th) {
  cv::Mat Q(9, 2, CV_32F);                        // quy ước xếp θ chốt ở 1.5 (ví dụ: xen kẽ)
  for (int i = 0; i < 9; ++i) { Q.at<float>(i, 0) = th[2 * i]; Q.at<float>(i, 1) = th[2 * i + 1]; }
  // nếu 1.5 cho thấy θ là độ lệch: cộng thêm tọa độ lưới đích vào Q ở đây
  cv::Mat G = p_->T * Q;                          // 12544 x 2, lưới nguồn trong [-1, 1]
  cv::Mat mx(112, 112, CV_32F), my(112, 112, CV_32F);
  const float sx = (roi224.cols - 1) * 0.5f, sy = (roi224.rows - 1) * 0.5f;   // align_corners = True
  for (int i = 0; i < 112 * 112; ++i) {
    mx.ptr<float>()[i] = (G.at<float>(i, 0) + 1.f) * sx;
    my.ptr<float>()[i] = (G.at<float>(i, 1) + 1.f) * sy;
  }
  cv::Mat out;
  cv::remap(rgb32(roi224), out, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
  return out;                                     // 112x112 CV_32FC3, RGB, thang 0-255
}

Eigen::VectorXf EmbeddingModel::embed(const cv::Mat& aligned112) {
  auto in = toNchw(aligned112);
  auto out = run(*p_->embed, in, {1, 3, 112, 112}, "x", "emb");
  Eigen::VectorXf v = Eigen::Map<Eigen::VectorXf>(out.data(), 512);
  return v.normalized();
}
```

Ba điểm dễ sai: thứ tự xếp 18 số của θ, θ là tuyệt đối hay độ lệch, và `align_corners`. Cả ba phải khớp kết luận ở bước 1.5. Vùng ngoài ảnh lấy giá trị 0, cùng với `grid_sample` mặc định.

### 3.2 Tham chiếu từ Python và test

Tạo tham chiếu bằng chính PyTorch, với cùng tiền xử lý như C++ (cùng `cv2.resize` kiểu `INTER_AREA`). Nhờ vậy mọi lệch số giữa C++ và tham chiếu là lỗi của C++.

```python
# python/export_cpp_ref.py
import glob, os, numpy as np, cv2, torch, torch.nn.functional as F
from matdag import MatDag, find_vars
from tps import tps_T

dag = MatDag('EE_PRnet-epoch-60.mat').eval()
tv, av, ev = find_vars(dag)
T = tps_T(112, 112)
chw = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None]
rgb = lambda m: cv2.cvtColor(m, cv2.COLOR_BGR2RGB).astype(np.float32)
os.makedirs('ref_cpp', exist_ok=True)
for i, f in enumerate(sorted(glob.glob('ref_in/*.png'))[:10]):
    bgr = cv2.imread(f)                                            # ROI 224x224
    small = cv2.resize(bgr, (56, 56), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        th = dag({'ANInput': chw(rgb(small))}, tv).flatten()
        grid = (T @ th.reshape(9, 2)).reshape(1, 112, 112, 2)      # quy ước chốt ở 1.5
        ali = F.grid_sample(chw(rgb(bgr)), grid, align_corners=True)
        emb = dag({av: ali}, ev).flatten()
    n = f'ref_cpp/{i:03d}'
    cv2.imwrite(n + '.png', bgr)
    th.numpy().tofile(n + '.theta.bin'); emb.numpy().tofile(n + '.emb.bin')
```

```cpp
// tests/test_embedding.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include "interfaces.h"

static std::vector<float> loadFloats(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  std::vector<float> v(static_cast<size_t>(f.tellg()) / sizeof(float));
  f.seekg(0); f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
  return v;
}

TEST_CASE("embedding C++ khớp tham chiếu Python") {
  EmbeddingModel m("models");
  for (const auto& e : std::filesystem::directory_iterator("ref_cpp")) {
    if (e.path().extension() != ".png") continue;
    auto stem = e.path(); stem.replace_extension("");
    const std::string base = stem.string();
    cv::Mat roi = cv::imread(base + ".png");
    auto th = m.regressTheta(roi);
    auto thRef = loadFloats(base + ".theta.bin");
    for (int i = 0; i < 18; ++i) CHECK(th[i] == Catch::Approx(thRef[i]).margin(1e-3));
    auto embRef = loadFloats(base + ".emb.bin");
    Eigen::VectorXf ref = Eigen::Map<Eigen::VectorXf>(embRef.data(), 512);
    CHECK(m.embed(m.warpTps(roi, th)).dot(ref.normalized()) > 0.9999f);
  }
}
```

Ngoài test số, hãy lưu ảnh warp của vài mẫu ra PNG và xem bằng mắt: ROI phải trông như một lòng bàn tay không bị méo.

**Nghiệm thu:** test trên 10 ảnh tham chiếu đạt (θ lệch dưới 1e-3, cosine embedding trên 0.9999), ảnh warp quan sát được bình thường, và chạy đủ ba bước cho một ảnh mà không lỗi bộ nhớ. Ghi lại thời gian trung bình của từng bước để điền bảng ở giai đoạn 9.
