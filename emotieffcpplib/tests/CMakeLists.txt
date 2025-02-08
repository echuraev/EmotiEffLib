cmake_minimum_required(VERSION 3.10.2)

enable_testing()

set(MTCNN_PATH "${PROJECT_SOURCE_DIR}/3rdparty/opencv-mtcnn")
add_subdirectory("${MTCNN_PATH}" "${CMAKE_BINARY_DIR}/3rdparty/opencv-mtcnn")

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/3rdparty/opencv-mtcnn/lib/include"
)

file(GLOB src_files
    "*.cpp"
)

add_executable(unit_tests
  ${src_files}
)
target_link_libraries(
  unit_tests
  ${PROJECT_NAME}
  GTest::gtest_main
  mtcnn
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
