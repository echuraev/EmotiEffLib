cmake_minimum_required(VERSION 3.10.2)
project("emotiefflib")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenCV REQUIRED)

option(BUILD_TESTS "Build EmotiEffCppLib with tests" OFF)
option(WITH_TORCH "Path to the direcotry with Libtorch" OFF)
option(WITH_ONNX "Build EmotiEffCppLib with tests" OFF)

if(NOT WITH_TORCH AND NOT WITH_ONNX)
    message(FATAL_ERROR "At least one inference engine should be specified: -DWITH_TORCH or -DWITH_ONNX")
endif()

if (WITH_TORCH)
    list(APPEND CMAKE_PREFIX_PATH "${WITH_TORCH}")
    find_package(Torch REQUIRED)
endif()

include_directories(
    "${PROJECT_SOURCE_DIR}/include"
)

file(GLOB src_files
    "src/*.cpp"
)

add_library(${PROJECT_NAME} STATIC
    ${src_files}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS} # ??? Is it really necessary
    ${TORCH_LIBRARIES}
)

if(${BUILD_TESTS})
    include(FetchContent)
    FetchContent_Declare(
      googletest
      # Release v1.16.0
      URL https://github.com/google/googletest/archive/6910c9d9165801d8827d628cb72eb7ea9dd538c5.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
# For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()
