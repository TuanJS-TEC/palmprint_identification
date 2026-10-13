cmake_minimum_required(VERSION 3.18)
project(palm_sdk VERSION 1.0.0 LANGUAGES C CXX)
# Palm SDK: right-hand palmprint registration / login from camera frames.
#
#   libpalm.so    shared library exporting only the C API of include/palm_api.h
#   libpalm_static.a   the same code as a static library (link it into a JNI .so)
#   palm_cli, test_golden   desktop / adb tools (PALM_BUILD_TOOLS, need OpenCV imgcodecs)
#
# Desktop:  cmake -S . -B build && cmake --build build
# Android:  see README.md (NDK toolchain, OpenCV Android SDK, onnxruntime-android AAR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)  # libpalm.so exports the PALM_API functions only
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(ANDROID)
  option(PALM_BUILD_TOOLS "build palm_cli and test_golden (adb shell executables)" OFF)
else()
  option(PALM_BUILD_TOOLS "build palm_cli and test_golden" ON)
endif()

set(PALM_OPENCV_COMPONENTS core imgproc)
if(PALM_BUILD_TOOLS)
  list(APPEND PALM_OPENCV_COMPONENTS imgcodecs)
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS ${PALM_OPENCV_COMPONENTS})

# ONNX Runtime has no CMake package: ORT_ROOT is either a release directory
# (include/, lib/) or an unpacked onnxruntime-android AAR (headers/, jni/<abi>/).
if(ANDROID)
  set(ORT_ROOT "" CACHE PATH "unpacked onnxruntime-android AAR")
else()
  set(ORT_ROOT "/usr/local" CACHE PATH "ONNX Runtime install prefix")
endif()
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ORT_ROOT}/headers ${ORT_ROOT}/include ${ORT_ROOT}/include/onnxruntime
        ${ORT_ROOT}/include/onnxruntime/core/session
  NO_CMAKE_FIND_ROOT_PATH REQUIRED)
find_library(ORT_LIBRARY onnxruntime
  HINTS ${ORT_ROOT}/jni/${ANDROID_ABI} ${ORT_ROOT}/lib
  NO_CMAKE_FIND_ROOT_PATH REQUIRED)
message(STATUS "ONNX Runtime: ${ORT_LIBRARY}")

# SQLite for the gallery database (amalgamation in third_party/): built in, with
# hidden symbols, so it cannot clash with a SQLite of the app.
add_library(sqlite3_palm STATIC third_party/sqlite3/sqlite3.c)
target_include_directories(sqlite3_palm PUBLIC third_party/sqlite3)
target_compile_definitions(sqlite3_palm PRIVATE
  SQLITE_OMIT_LOAD_EXTENSION SQLITE_THREADSAFE=1 SQLITE_DQS=0 SQLITE_DEFAULT_MEMSTATUS=0)
set_target_properties(sqlite3_palm PROPERTIES C_VISIBILITY_PRESET hidden POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(sqlite3_palm PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

set(PALM_SOURCES
  src/config.cpp
  src/gallery.cpp
  src/hand_crop.cpp
  src/hand_detector.cpp
  src/palm_api.cpp
  src/palm_model.cpp
  src/preprocess.cpp)
add_library(palm_objects OBJECT ${PALM_SOURCES})
target_include_directories(palm_objects PUBLIC include src ${ORT_INCLUDE_DIR})
target_link_libraries(palm_objects PUBLIC opencv_core opencv_imgproc sqlite3_palm)  # include directories
if(NOT MSVC)
  target_compile_options(palm_objects PRIVATE -Wall -Wextra)
endif()

add_library(palm SHARED $<TARGET_OBJECTS:palm_objects>)
target_include_directories(palm PUBLIC include)
target_link_libraries(palm PRIVATE opencv_core opencv_imgproc ${ORT_LIBRARY} sqlite3_palm)
if(NOT APPLE AND NOT MSVC)
  # exports: palm_* only (no C++ template instances, no symbols of a static OpenCV)
  target_link_options(palm PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/palm.map" "-Wl,--exclude-libs,ALL")
  set_target_properties(palm PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/palm.map)
endif()
if(ANDROID)
  target_link_options(palm PRIVATE "-Wl,-z,max-page-size=16384")  # 16 KB page devices (Android 15+)
endif()

add_library(palm_static STATIC $<TARGET_OBJECTS:palm_objects>)
target_include_directories(palm_static PUBLIC include src ${ORT_INCLUDE_DIR})
target_link_libraries(palm_static PUBLIC opencv_core opencv_imgproc ${ORT_LIBRARY} sqlite3_palm)

if(PALM_BUILD_TOOLS)
  add_executable(palm_cli tools/palm_cli.cpp)  # uses internal headers: links the static library
  target_link_libraries(palm_cli PRIVATE palm_static opencv_imgcodecs)
  add_executable(test_golden tests/test_golden.cpp)  # C API only: links libpalm.so
  target_link_libraries(test_golden PRIVATE palm opencv_core opencv_imgproc opencv_imgcodecs)
  set_target_properties(palm_cli test_golden PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")

  enable_testing()
  add_test(NAME golden COMMAND test_golden ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_CURRENT_SOURCE_DIR}/testdata)
endif()
