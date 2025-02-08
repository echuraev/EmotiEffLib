cmake_minimum_required(VERSION 3.10.2)

enable_testing()

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
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
