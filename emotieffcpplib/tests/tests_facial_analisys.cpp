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

class EmotiEffLibOnlyModelTests : public ::testing::TestWithParam<std::string> {};

TEST_P(EmotiEffLibTests, OneImagePredictionOneModel) {
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
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, modelPath);
    for (auto& face : facialImages) {
        auto res = fer->precictEmotions(face, true);
        emotions.push_back(res.labels[0]);
    }

    ASSERT_TRUE(AreVectorsEqual(emotions, expEmotions));

    // Try to call unsuitable functions
    try {
        fer->extractFeatures(facialImages[0]);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("Model for features extraction wasn't specified in the config!",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
    try {
        xt::xarray<float> tmp;
        fer->classifyEmotions(tmp);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("Model for emotions classification wasn't specified in the config!",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
}

TEST_P(EmotiEffLibTests, OneImagePredictionTwoModels) {
    auto& [backend, modelName] = GetParam();
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path imagePath(pyTestDir);
    imagePath = imagePath / "test_images" / "20180720_174416.jpg";
    cv::Mat frame = cv::imread(imagePath);
    auto facialImages = recognizeFaces(frame);

    fs::path modelPath(getEmotiEffLibRootDir());
    std::string ext = (backend == "torch") ? ".pt" : ".onnx";
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    std::string featureExtractorPath = modelPath / ("features_extractor_" + modelName + ext);
    std::string classifierPath = modelPath / ("classifier_" + modelName + ext);
    EmotiEffLib::EmotiEffLibConfig config = {
        .backend = backend,
        .featureExtractorPath = featureExtractorPath,
        .classifierPath = classifierPath,
        .modelName = modelName,
    };
    std::vector<std::string> expEmotions;
    if (modelName == "enet_b0_8_va_mtl" ||
        (backend == "onnx" && modelName == "enet_b0_8_best_afew")) {
        expEmotions = {"Anger", "Happiness", "Happiness"};
    } else {
        expEmotions = {"Anger", "Happiness", "Fear"};
    }
    std::vector<std::string> emotions;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    for (auto& face : facialImages) {
        auto res = fer->precictEmotions(face, true);
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

TEST_P(EmotiEffLibOnlyModelTests, OneImageFeatures) {
    std::string modelName = GetParam();
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path imagePath(pyTestDir);
    imagePath = imagePath / "test_images" / "20180720_174416.jpg";
    cv::Mat frame = cv::imread(imagePath);
    auto facialImages = recognizeFaces(frame);

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    std::string featureExtractorOnnxPath =
        modelPath / ("features_extractor_" + modelName + ".onnx");
    std::string featureExtractorTorchPath = modelPath / ("features_extractor_" + modelName + ".pt");
    EmotiEffLib::EmotiEffLibConfig configOnnx = {
        .backend = "onnx",
        .featureExtractorPath = featureExtractorOnnxPath,
    };
    EmotiEffLib::EmotiEffLibConfig configTorch = {
        .backend = "torch",
        .featureExtractorPath = featureExtractorTorchPath,
    };
    auto ferOnnx = EmotiEffLib::EmotiEffLibRecognizer::createInstance(configOnnx);
    auto ferTorch = EmotiEffLib::EmotiEffLibRecognizer::createInstance(configTorch);
    for (auto& face : facialImages) {
        auto featuresOnnx = ferOnnx->extractFeatures(face);
        auto featuresTorch = ferTorch->extractFeatures(face);
        EXPECT_EQ(featuresOnnx.shape(), featuresTorch.shape());
    }
}

std::string OnlyModelTestNameGenerator(
    const ::testing::TestParamInfo<EmotiEffLibOnlyModelTests::ParamType>& info) {
    auto modelName = info.param;
    std::ostringstream name;
    name << "model_" << modelName;

    // Replace invalid characters for test names
    std::string name_str = name.str();
    std::replace(name_str.begin(), name_str.end(), '.', '_'); // Replace dots
    return name_str;
}

INSTANTIATE_TEST_SUITE_P(FeaturesExtraction, EmotiEffLibOnlyModelTests,
                         ::testing::ValuesIn(EmotiEffLib::getSupportedModels()),
                         OnlyModelTestNameGenerator);

TEST(EmotiEffLibTests, CheckUnsupportedBackend) {
    try {
        EmotiEffLib::EmotiEffLibRecognizer::createInstance("OpenVINO", "my_model");
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("This backend (OpenVINO) is not supported. Please check your EmotiEffLib build "
                  "or configuration.",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
}

TEST(EmotiEffLibTests, CheckIncorrectConfig) {
    EmotiEffLib::EmotiEffLibConfig config;
    try {
        EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("This backend () is not supported. Please check your EmotiEffLib build or "
                  "configuration.",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
    config = {.backend = "torch", .classifierPath = "bla-bla", .modelName = "bla-bla"};
    try {
        EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("featureExtractorPath MUST be specified in the EmotiEffLibConfig.",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
    config.backend = "onnx";
    try {
        EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
        FAIL();
    } catch (const std::runtime_error& e) {
        EXPECT_EQ("featureExtractorPath MUST be specified in the EmotiEffLibConfig.",
                  std::string(e.what()));
    } catch (...) {
        FAIL();
    }
}
