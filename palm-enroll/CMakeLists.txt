cmake_minimum_required(VERSION 3.18)
project(palm_enroll CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)  # installed binaries keep finding OpenCV / ORT

# ONNX Runtime ships no CMake config in the release tarball: point ORT_ROOT at
# the unpacked onnxruntime-<os>-<ver> directory (default: /usr/local).
set(ORT_ROOT "/usr/local" CACHE PATH "ONNX Runtime install prefix")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ORT_ROOT}/include ${ORT_ROOT}/include/onnxruntime ${ORT_ROOT}/include/onnxruntime/core/session REQUIRED)
find_library(ORT_LIBRARY onnxruntime HINTS ${ORT_ROOT}/lib REQUIRED)
message(STATUS "ONNX Runtime: ${ORT_LIBRARY}")

add_library(palm
  src/preprocess.cpp
  src/palm_model.cpp
  src/gallery.cpp
  src/config.cpp
  src/hand_detector.cpp
  src/hand_crop.cpp
  src/palm_api.cpp)
target_include_directories(palm PUBLIC src ${ORT_INCLUDE_DIR})
target_link_libraries(palm PUBLIC ${OpenCV_LIBS} ${ORT_LIBRARY})
if(MSVC)
  target_compile_options(palm PRIVATE /W4)
else()
  target_compile_options(palm PRIVATE -Wall -Wextra)
endif()

add_executable(palm_cli tools/palm_cli.cpp)
target_link_libraries(palm_cli PRIVATE palm)
add_executable(palm_eval tools/palm_eval.cpp)
target_link_libraries(palm_eval PRIVATE palm)
add_executable(palm_handcheck tools/palm_handcheck.cpp)
target_link_libraries(palm_handcheck PRIVATE palm)
add_executable(palm_bench tools/palm_bench.cpp)
target_link_libraries(palm_bench PRIVATE palm)
add_executable(palm_profile tools/palm_profile.cpp)
target_link_libraries(palm_profile PRIVATE palm)
add_executable(palm_soak tools/palm_soak.cpp)  # replaces malloc & co. to count allocations
target_link_libraries(palm_soak PRIVATE palm)

# Qt5 GUI with a live camera (optional)
find_package(Qt5 COMPONENTS Widgets QUIET)
if(Qt5Widgets_FOUND)
  add_executable(palm_gui gui/main.cpp gui/main_window.cpp gui/camera_worker.cpp gui/video_view.cpp gui/event_log.cpp)
  set_target_properties(palm_gui PROPERTIES AUTOMOC ON)
  target_link_libraries(palm_gui PRIVATE palm Qt5::Widgets)
else()
  message(STATUS "Qt5 Widgets not found: palm_gui is not built")
endif()

# ---------------------------------------------------------------- deploy
# cmake --install build --prefix ../binary : a folder that runs on its own
# (executables, config.yaml with paths into models/, the ONNX models).
# gallery.bin in the prefix is left alone; config.yaml is overwritten.
set(PALM_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../models" CACHE PATH "folder with the ONNX models")
install(TARGETS palm_cli DESTINATION .)
if(TARGET palm_gui)
  install(TARGETS palm_gui DESTINATION .)
endif()
install(FILES
  ${PALM_MODELS_DIR}/EE_PRnet-epoch-60.onnx
  ${PALM_MODELS_DIR}/palm_detection_mediapipe_2023feb.onnx
  ${PALM_MODELS_DIR}/handpose_estimation_mediapipe_2023feb.onnx
  ${PALM_MODELS_DIR}/LICENSE.mediapipe-hand-models
  DESTINATION models)
install(PROGRAMS deploy/run_gui.sh DESTINATION .)
install(FILES deploy/eglfs_kms.json deploy/README.txt DESTINATION .)
install(CODE "
  file(READ \"${CMAKE_CURRENT_SOURCE_DIR}/config.yaml\" cfg)
  string(REPLACE \"../models/\" \"models/\" cfg \"\${cfg}\")
  file(WRITE \"\${CMAKE_INSTALL_PREFIX}/config.yaml\" \"\${cfg}\")
  message(STATUS \"Installing: \${CMAKE_INSTALL_PREFIX}/config.yaml\")
")

# ---------------------------------------------------------------- tests
enable_testing()
set(PALM_ONNX "${CMAKE_CURRENT_SOURCE_DIR}/../models/EE_PRnet-epoch-60.onnx" CACHE FILEPATH "EE-PRnet ONNX for tests")

add_executable(test_preprocess tests/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE palm)
add_test(NAME preprocess COMMAND test_preprocess)

add_executable(test_gallery tests/test_gallery.cpp)
target_link_libraries(test_gallery PRIVATE palm)
add_test(NAME gallery COMMAND test_gallery)

# needs tests/data/ref from python/export_cpp_ref.py; exits 77 (skipped) when missing
add_executable(test_parity tests/test_parity.cpp)
target_link_libraries(test_parity PRIVATE palm)
add_test(NAME parity COMMAND test_parity ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/ref ${PALM_ONNX})
set_tests_properties(parity PROPERTIES SKIP_RETURN_CODE 77)

# C API end to end on the capture log frames; exits 77 (skipped) when the models or frames are missing
add_executable(test_api tests/test_api.cpp)
target_link_libraries(test_api PRIVATE palm)
add_test(NAME api COMMAND test_api ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml
         ${CMAKE_CURRENT_SOURCE_DIR}/../log_label.csv ${CMAKE_CURRENT_SOURCE_DIR}/../output/original)
set_tests_properties(api PROPERTIES SKIP_RETURN_CODE 77)
