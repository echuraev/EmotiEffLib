#include "test_utils.h"
#include "gtest/gtest.h"
#include <emotiefflib/facial_analysis.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

namespace fs = std::filesystem;

using EmotiEffLibTestParams = std::tuple<std::string, std::string>;

class EmotiEffLibTests : public ::testing::TestWithParam<EmotiEffLibTestParams> {};

TEST_P(EmotiEffLibTests, OneImagePrediction) {
    auto& [backend, modelName] = GetParam();
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path imagePath(pyTestDir);
    imagePath = imagePath / "test_images" / "20180720_174416.jpg";
    cv::Mat frame = cv::imread(imagePath);
    auto facialImages = recognizeFaces(frame);

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    if (backend == "torch") {
        modelPath /= modelName + ".pt";
    } else {
        modelPath /= modelName + ".onnx";
    }
    std::vector<std::string> expEmotions;
    if (modelName == "enet_b0_8_va_mtl" ||
        (backend == "onnx" && modelName == "enet_b0_8_best_afew")) {
        expEmotions = {"Anger", "Happiness", "Happiness"};
    } else {
        expEmotions = {"Anger", "Happiness", "Fear"};
    }
    std::vector<std::string> emotions;
    for (auto& face : facialImages) {
        auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, modelPath);
        auto res = fer->precictEmotions(face, false);
        emotions.push_back(res.labels[0]);
    }

    ASSERT_TRUE(AreVectorsEqual(emotions, expEmotions));
}

std::string TestNameGenerator(const ::testing::TestParamInfo<EmotiEffLibTests::ParamType>& info) {
    auto& [backend, modelName] = info.param;
    std::ostringstream name;
    name << "backend_" << backend << "_model_" << modelName;

    // Replace invalid characters for test names
    std::string name_str = name.str();
    std::replace(name_str.begin(), name_str.end(), '.', '_'); // Replace dots
    return name_str;
}

INSTANTIATE_TEST_SUITE_P(
    Emotions, EmotiEffLibTests,
    ::testing::Combine(::testing::ValuesIn(EmotiEffLib::getAvailableBackends()),
                       ::testing::ValuesIn(EmotiEffLib::getSupportedModels())),
    TestNameGenerator);
