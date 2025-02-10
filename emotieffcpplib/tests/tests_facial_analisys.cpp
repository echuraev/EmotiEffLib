#include "test_utils.h"
#include <emotiefflib/facial_analysis.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

namespace fs = std::filesystem;

class EmotiEffLibTests : public ::testing::TestWithParam<std::string> {};

TEST_P(EmotiEffLibTests, Basic) {
    std::string backend = GetParam();
    std::cout << "Backend: " << backend << std::endl;
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path imagePath(pyTestDir);
    imagePath = imagePath / "test_images" / "20180720_174416.jpg";
    cv::Mat frame = cv::imread(imagePath);
    auto facialImages = recognizeFaces(frame);

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    if (backend == "torch") {
        modelPath /= "enet_b0_8_best_vgaf.pt";
    } else {
        modelPath /= "enet_b0_8_best_vgaf.onnx";
    }
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, modelPath);
}

INSTANTIATE_TEST_SUITE_P(Basic, EmotiEffLibTests,
                         ::testing::ValuesIn(EmotiEffLib::getAvailableBackends()));
