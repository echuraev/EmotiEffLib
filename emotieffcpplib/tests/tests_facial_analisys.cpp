#include "test_utils.h"
#include <emotiefflib/facial_analysis.h>
#include <filesystem>
#include <gtest/gtest.h>

namespace fs = std::filesystem;

TEST(EmotionRecognition, Basic) {
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path imagePath(pyTestDir);
    imagePath = imagePath / "test_images" / "20180720_174416.jpg";
    cv::Mat frame = cv::imread(imagePath);
    auto facialImages = recognizeFaces(frame);
    EmotiEffLib::EmotiEffLibRecognizerTorch("asdfsf", "sdfsdf");
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}
