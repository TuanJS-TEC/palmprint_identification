cmake_minimum_required(VERSION 3.18)
project(palm_enroll C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Built on its own this is the core library + palm_cli; another project uses it
# with add_subdirectory(<path>/palm-enroll) and links palm::palm.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(PALM_IS_TOP_LEVEL ON)
else()
  set(PALM_IS_TOP_LEVEL OFF)
endif()
option(PALM_BUILD_CLI "Build the palm_cli command line tool" ${PALM_IS_TOP_LEVEL})

if(PALM_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)  # installed binaries keep finding OpenCV / ORT

# ONNX Runtime ships no CMake config in the release tarball: point ORT_ROOT at
# the unpacked onnxruntime-<os>-<ver> directory (default: /usr/local).
if(APPLE AND EXISTS /opt/homebrew/opt/onnxruntime)
  set(ORT_DEFAULT_ROOT /opt/homebrew/opt/onnxruntime)  # brew install onnxruntime
else()
  set(ORT_DEFAULT_ROOT /usr/local)
endif()
set(ORT_ROOT "${ORT_DEFAULT_ROOT}" CACHE PATH "ONNX Runtime install prefix")
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ORT_ROOT}/include ${ORT_ROOT}/include/onnxruntime ${ORT_ROOT}/include/onnxruntime/core/session REQUIRED)
find_library(ORT_LIBRARY onnxruntime HINTS ${ORT_ROOT}/lib REQUIRED)
message(STATUS "ONNX Runtime: ${ORT_LIBRARY}")

# ------------------------------------------------------------ third party
# SQLite (amalgamation in third_party/): the gallery database. Built in, so no
# system SQLite is needed; its symbols stay hidden. OBJECT: its objects go into
# libpalm.a, so the SDK is one archive.
add_library(sqlite3_palm OBJECT third_party/sqlite3/sqlite3.c)
target_include_directories(sqlite3_palm PUBLIC third_party/sqlite3)
target_compile_definitions(sqlite3_palm PRIVATE
  SQLITE_OMIT_LOAD_EXTENSION SQLITE_THREADSAFE=1 SQLITE_DQS=0 SQLITE_DEFAULT_MEMSTATUS=0)
set_target_properties(sqlite3_palm PROPERTIES C_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(sqlite3_palm PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

# ------------------------------------------------------------ core library
#   common       Vec, cosine
#   recognition  EE-PRnet: vl_imreadjpeg-style resize + ONNX model
#   detection    MediaPipe hand detector + capture gate (is this frame good enough?)
#   gallery      enrollment policy (template), enrolled people, SQLite storage
#   config       config.yaml -> palm::Config
#   engine       C API (palm/engine.h) over all of the above
# Public headers: include/palm/ (palm/palm.h includes all C++ ones). Private ones sit next to the sources.
add_library(palm STATIC
  src/common/vec.cpp
  src/recognition/preprocess.cpp
  src/recognition/palm_model.cpp
  src/detection/hand_detector.cpp
  src/detection/capture_gate.cpp
  src/gallery/template.cpp
  src/gallery/gallery.cpp
  src/gallery/gallery_sqlite.cpp
  src/config/config.cpp
  src/engine/engine.cpp)
add_library(palm::palm ALIAS palm)
set_target_properties(palm PROPERTIES POSITION_INDEPENDENT_CODE ON)  # the SDK user may link it into a .so
target_include_directories(palm
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE src/recognition src/gallery ${ORT_INCLUDE_DIR})
# OpenCV types are in the public headers; ONNX Runtime and SQLite are not.
target_link_libraries(palm PUBLIC ${OpenCV_LIBS} PRIVATE ${ORT_LIBRARY} sqlite3_palm)
if(MSVC)
  target_compile_options(palm PRIVATE /W4)
else()
  target_compile_options(palm PRIVATE -Wall -Wextra)
endif()

# ------------------------------------------------------------ command line tool
if(PALM_BUILD_CLI)
  add_executable(palm_cli apps/palm_cli/palm_cli.cpp)
  target_link_libraries(palm_cli PRIVATE palm::palm)
endif()

# ------------------------------------------------------------ C API self-check
# ./build/palm_engine_test ../models   (C source: proves palm/engine.h is plain C)
if(PALM_IS_TOP_LEVEL)
  add_executable(palm_engine_test tests/engine_test.c)
  target_link_libraries(palm_engine_test PRIVATE palm::palm)
endif()

# ---------------------------------------------------------------- SDK
# cmake --install build --component sdk --prefix <dir> : the folder sent to the
# C API user (packaging/build_linux_sdk.sh builds it for Linux x86_64).
# EXCLUDE_FROM_ALL: the plain install into ../binary stays as it was.
# config.yaml with "../models/" -> "models/": both installs put the models there.
set(PALM_INSTALL_CONFIG "
  file(READ \"${CMAKE_CURRENT_SOURCE_DIR}/config.yaml\" cfg)
  string(REPLACE \"../models/\" \"models/\" cfg \"\${cfg}\")
  file(WRITE \"\${CMAKE_INSTALL_PREFIX}/config.yaml\" \"\${cfg}\")
  message(STATUS \"Installing: \${CMAKE_INSTALL_PREFIX}/config.yaml\")
")
install(TARGETS palm ARCHIVE DESTINATION lib COMPONENT sdk EXCLUDE_FROM_ALL)
install(FILES include/palm/engine.h DESTINATION include/palm COMPONENT sdk EXCLUDE_FROM_ALL)
install(FILES tests/engine_test.c DESTINATION examples COMPONENT sdk EXCLUDE_FROM_ALL)
install(FILES packaging/README.txt packaging/Makefile DESTINATION . COMPONENT sdk EXCLUDE_FROM_ALL)
install(CODE "${PALM_INSTALL_CONFIG}" COMPONENT sdk EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------- deploy
# cmake --install build --prefix ../binary : a folder that runs on its own
# (palm_cli, config.yaml with paths into models/, the ONNX models).
# gallery.db in the prefix is left alone; config.yaml is overwritten.
if(PALM_BUILD_CLI)
  set(PALM_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../models" CACHE PATH "folder with the ONNX models")
  install(TARGETS palm_cli DESTINATION .)
  install(FILES
    ${PALM_MODELS_DIR}/EE_PRnet-epoch-60.onnx
    ${PALM_MODELS_DIR}/palm_detection_mediapipe_2023feb.onnx
    ${PALM_MODELS_DIR}/handpose_estimation_mediapipe_2023feb.onnx
    DESTINATION models)
  if(EXISTS ${PALM_MODELS_DIR}/LICENSE.mediapipe-hand-models)
    install(FILES ${PALM_MODELS_DIR}/LICENSE.mediapipe-hand-models DESTINATION models)
  endif()
  install(CODE "${PALM_INSTALL_CONFIG}")
endif()
