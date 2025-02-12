#include "test_utils.h"
#include "gtest/gtest.h"
#include <emotiefflib/facial_analysis.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>

namespace fs = std::filesystem;

namespace {
std::vector<std::string> getOneImageExpEmotions(const std::string& backend,
                                                const std::string& modelName) {
    if (modelName == "enet_b0_8_va_mtl" ||
        (backend == "onnx" && modelName == "enet_b0_8_best_afew")) {
        return {"Anger", "Happiness", "Happiness"};
    }
    return {"Anger", "Happiness", "Fear"};
}

std::vector<cv::Mat> getOneImageFaces() {
    std::string pyTestDir = getPathToPythonTestDir();
    fs::path imagePath(pyTestDir);
    imagePath = imagePath / "test_images" / "20180720_174416.jpg";
    cv::Mat frame = cv::imread(imagePath);
    return recognizeFaces(frame);
}
} // namespace

using EmotiEffLibTestParams = std::tuple<std::string, std::string>;

class EmotiEffLibTests : public ::testing::TestWithParam<EmotiEffLibTestParams> {};

class EmotiEffLibOnlyModelTests : public ::testing::TestWithParam<std::string> {};

TEST_P(EmotiEffLibTests, OneImagePredictionOneModel) {
    auto& [backend, modelName] = GetParam();
    auto facialImages = getOneImageFaces();

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    if (backend == "torch") {
        modelPath /= modelName + ".pt";
    } else {
        modelPath /= modelName + ".onnx";
    }
    std::vector<std::string> emotions;
    std::vector<std::string> scorePrediction;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, modelPath);
    for (auto& face : facialImages) {
        auto res = fer->predictEmotions(face, true);
        emotions.push_back(res.labels[0]);
        auto pred = xt::argmax(res.scores, 1);
        scorePrediction.push_back(fer->getEmotionClassById(pred[0]));
    }

    ASSERT_TRUE(AreVectorsEqual(emotions, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(emotions, getOneImageExpEmotions(backend, modelName)));

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

TEST_P(EmotiEffLibTests, OneImageMultiPredictionOneModel) {
    auto& [backend, modelName] = GetParam();
    auto facialImages = getOneImageFaces();

    fs::path modelPath(getEmotiEffLibRootDir());
    modelPath = modelPath / "models" / "emotieffcpplib_prepared_models";
    if (backend == "torch") {
        modelPath /= modelName + ".pt";
    } else {
        modelPath /= modelName + ".onnx";
    }
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, modelPath);
    auto result = fer->predictEmotions(facialImages, true);
    auto preds = xt::argmax(result.scores, 1);

    std::vector<std::string> scorePrediction;
    for (auto& pred : preds) {
        scorePrediction.push_back(fer->getEmotionClassById(pred));
    }

    ASSERT_TRUE(AreVectorsEqual(result.labels, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(result.labels, getOneImageExpEmotions(backend, modelName)));

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
    auto facialImages = getOneImageFaces();

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
    std::vector<std::string> emotions;
    std::vector<std::string> scorePrediction;
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    for (auto& face : facialImages) {
        auto res = fer->predictEmotions(face, true);
        emotions.push_back(res.labels[0]);
        auto pred = xt::argmax(res.scores, 1);
        scorePrediction.push_back(fer->getEmotionClassById(pred[0]));
    }

    ASSERT_TRUE(AreVectorsEqual(emotions, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(emotions, getOneImageExpEmotions(backend, modelName)));
}

TEST_P(EmotiEffLibTests, OneImageMultiPredictionTwoModels) {
    auto& [backend, modelName] = GetParam();
    auto facialImages = getOneImageFaces();

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
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    auto result = fer->predictEmotions(facialImages, true);
    auto preds = xt::argmax(result.scores, 1);

    std::vector<std::string> scorePrediction;
    for (auto& pred : preds) {
        scorePrediction.push_back(fer->getEmotionClassById(pred));
    }

    ASSERT_TRUE(AreVectorsEqual(result.labels, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(result.labels, getOneImageExpEmotions(backend, modelName)));
}

TEST_P(EmotiEffLibTests, OneImageClassification) {
    auto& [backend, modelName] = GetParam();
    auto facialImages = getOneImageFaces();

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
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    std::vector<std::string> emotions;
    std::vector<std::string> scorePrediction;
    for (auto& face : facialImages) {
        auto features = fer->extractFeatures(face);
        auto res = fer->classifyEmotions(features);
        emotions.push_back(res.labels[0]);
        auto pred = xt::argmax(res.scores, 1);
        scorePrediction.push_back(fer->getEmotionClassById(pred[0]));
    }

    ASSERT_TRUE(AreVectorsEqual(emotions, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(emotions, getOneImageExpEmotions(backend, modelName)));
}

TEST_P(EmotiEffLibTests, OneImageMultiClassification) {
    auto& [backend, modelName] = GetParam();
    auto facialImages = getOneImageFaces();

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
    auto fer = EmotiEffLib::EmotiEffLibRecognizer::createInstance(config);
    auto features = fer->extractFeatures(facialImages);
    auto result = fer->classifyEmotions(features);
    auto preds = xt::argmax(result.scores, 1);

    std::vector<std::string> scorePrediction;
    for (auto& pred : preds) {
        scorePrediction.push_back(fer->getEmotionClassById(pred));
    }

    ASSERT_TRUE(AreVectorsEqual(result.labels, scorePrediction));
    ASSERT_TRUE(AreVectorsEqual(result.labels, getOneImageExpEmotions(backend, modelName)));
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
    auto facialImages = getOneImageFaces();

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

TEST_P(EmotiEffLibOnlyModelTests, OneImageMultiFeatures) {
    std::string modelName = GetParam();
    auto facialImages = getOneImageFaces();

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
    auto featuresOnnx = ferOnnx->extractFeatures(facialImages);
    auto featuresTorch = ferTorch->extractFeatures(facialImages);
    EXPECT_EQ(featuresOnnx.shape()[0], 3);
    EXPECT_EQ(featuresOnnx.shape(), featuresTorch.shape());
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
