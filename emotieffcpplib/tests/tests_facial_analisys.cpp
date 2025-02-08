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

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "affectnet_emotions" / "enet_b0_8_best_vgaf.pt";
    EmotiEffLib::EmotiEffLibRecognizerTorch(modelPath, "sdfsdf");
}
