#!/bin/sh
# Assembles the palm SDK handed to the mobile team:
#   OUT_DIR/palm-sdk-YYYYMMDD/  and  OUT_DIR/palm-sdk-YYYYMMDD.zip
# then builds the copy on this machine, records its golden test data from the
# capture log frames and runs the golden test on it.
# usage: palm-enroll/sdk/make_sdk.sh [OUT_DIR]     (default: DMS_palmprint/dist)
set -eu
here=$(cd "$(dirname "$0")" && pwd)  # palm-enroll/sdk
root=$(dirname "$here")              # palm-enroll
proj=$(dirname "$root")              # DMS_palmprint
out=${1:-$proj/dist}
name=palm-sdk-$(date +%Y%m%d)
pkg=$out/$name
frames=$proj/output/original         # mirror images of the camera view

rm -rf "$pkg"
mkdir -p "$pkg/include" "$pkg/src" "$pkg/tools" "$pkg/tests" "$pkg/models" "$pkg/testdata" "$pkg/third_party/sqlite3"
cp "$here/CMakeLists.txt" "$here/palm.map" "$here/README.md" "$here/NOTICE.md" "$pkg/"
cp "$root/src/palm_api.h" "$pkg/include/"
cp "$root/src/palm_api.cpp" "$pkg/src/"
for f in config gallery hand_crop hand_detector palm_model preprocess; do
  cp "$root/src/$f.cpp" "$root/src/$f.h" "$pkg/src/"
done
cp "$root/third_party/sqlite3/sqlite3.c" "$root/third_party/sqlite3/sqlite3.h" "$root/third_party/sqlite3/README.txt" "$pkg/third_party/sqlite3/"
cp "$root/tools/palm_cli.cpp" "$root/tools/args.h" "$pkg/tools/"
cp "$here/test_golden.cpp" "$pkg/tests/"
for m in EE_PRnet-epoch-60.onnx palm_detection_mediapipe_2023feb.onnx \
         handpose_estimation_mediapipe_2023feb.onnx LICENSE.mediapipe-hand-models; do
  cp "$proj/models/$m" "$pkg/models/"
done
(cd "$pkg/models" && sha256sum ./*.onnx > SHA256SUMS)
sed -e 's#"\.\./models/#"models/#' "$root/config.yaml" > "$pkg/config.yaml"

# Golden data: person A (5 frames to register + 1 to log in), a stranger and a
# left hand, from output/original; recorded by the copy itself.
build=$out/build-$name
rm -rf "$build"
cmake -S "$pkg" -B "$build" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$build" -j4
"$build/test_golden" "$pkg/models" "$pkg/testdata" --generate \
  "$frames/palm_0001_20261001_151307_024_full.jpg" \
  "$frames/palm_0002_20261001_151310_917_full.jpg" \
  "$frames/palm_0003_20261001_151313_519_full.jpg" \
  "$frames/palm_0004_20261001_151314_518_full.jpg" \
  "$frames/palm_0005_20261001_151315_050_full.jpg" \
  "$frames/palm_0020_20261001_151324_014_full.jpg" \
  "$frames/palm_0070_20261001_152118_757_full.jpg" \
  "$frames/palm_0012_20261001_151847_183_full.jpg"
"$build/test_golden" "$pkg/models" "$pkg/testdata"
rm -rf "$build"

cd "$out"
rm -f "$name.zip"
python3 - "$name" << 'EOF'
import os, sys, zipfile
name = sys.argv[1]
with zipfile.ZipFile(name + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(name):
        dirs.sort()
        for f in sorted(files):
            z.write(os.path.join(d, f))
EOF
echo "SDK: $pkg"
echo "zip: $out/$name.zip ($(du -h "$name.zip" | cut -f1))"
