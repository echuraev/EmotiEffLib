#include "emotiefflib/backends/torch/facial_analysis.h"

#include <xtensor/xadapt.hpp>

namespace {
// Convert xt::xarray to torch::Tensor
torch::Tensor xarray2tensor(const xt::xarray<float>& xtensor) {
    std::vector<int64_t> shape(xtensor.shape().begin(), xtensor.shape().end());
    std::vector<float> data(xtensor.begin(), xtensor.end());

    return torch::from_blob(data.data(), shape, torch::TensorOptions().dtype(torch::kFloat))
        .clone();
}

// Convert torch::Tensor to xt::xarray
xt::xarray<float> tensor2xarray(const torch::Tensor& tensor) {
    torch::Tensor cpu_tensor =
        tensor.to(torch::kCPU).contiguous(); // Ensure CPU and contiguous memory

    // Get shape and data pointer
    std::vector<size_t> shape(cpu_tensor.sizes().begin(), cpu_tensor.sizes().end());
    const float* data_ptr = cpu_tensor.data_ptr<float>();

    // Adapt to xt::xarray
    return xt::adapt(data_ptr, cpu_tensor.numel(), xt::no_ownership(), shape);
    xt::xarray<float> result = xt::zeros<float>(shape);
    std::copy(data_ptr, data_ptr + tensor.numel(), result.begin());
    return result;
}
} // namespace

namespace EmotiEffLib {
EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& fullPipelineModel) {
    torch::jit::script::Module model = torch::jit::load(fullPipelineModel);
    model.to(torch::Device(torch::kCPU));
    model.eval();
    models_.push_back(model);
    fullPipelineModelIdx_ = 0;
    initRecognizer(fullPipelineModel);
}

EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const EmotiEffLibConfig& config) {
    configParser(config);
    initRecognizer(config.featureExtractorPath);
}

xt::xarray<float> EmotiEffLibRecognizerTorch::extractFeatures(const cv::Mat& faceImg) {
    if (featureExtractorIdx_ == -1)
        throw std::runtime_error("Model for features extraction wasn't specified in the config!");
    auto imgTensor = preprocess(faceImg);
    auto input = xarray2tensor(imgTensor);
    auto outputTensor = models_[featureExtractorIdx_].forward({input}).toTensor();
    auto features = tensor2xarray(outputTensor);
    return features;
}

EmotiEffLibRes EmotiEffLibRecognizerTorch::classifyEmotions(const xt::xarray<float>& features,
                                                            bool logits) {
    if (classifierIdx_ == -1)
        throw std::runtime_error(
            "Model for emotions classification wasn't specified in the config!");
    auto input = xarray2tensor(features);
    auto classifierOutput = models_[classifierIdx_].forward({input}).toTensor();
    auto scores = tensor2xarray(classifierOutput);
    return processScores(scores, logits);
}

EmotiEffLibRes EmotiEffLibRecognizerTorch::predictEmotions(const cv::Mat& faceImg, bool logits) {
    if (fullPipelineModelIdx_ == -1 && (featureExtractorIdx_ == -1 || classifierIdx_ == -1))
        throw std::runtime_error("predictEmotions method requires fillPipeline model or "
                                 "featureExtractor and classifier models");
    auto imgTensor = preprocess(faceImg);
    auto input = xarray2tensor(imgTensor);
    // Always in index 0 is the fullPipelineMode or featureExtractor model
    // In this case doesn't matter which one is here.
    auto outputTensor = models_[0].forward({input}).toTensor();
    xt::xarray<float> scores;
    if (fullPipelineModelIdx_ == -1 && classifierIdx_ > -1) {
        auto classifierOutput = models_[classifierIdx_].forward({outputTensor}).toTensor();
        scores = tensor2xarray(classifierOutput);
    } else {
        scores = tensor2xarray(outputTensor);
    }

    return processScores(scores, logits);
}

void EmotiEffLibRecognizerTorch::initRecognizer(const std::string& modelPath) {
    EmotiEffLibRecognizer::initRecognizer(modelPath);

    bool isB0 = modelName_.find("_b0_") != std::string::npos;
    imgSize_ = isB0 ? 224 : 260;
}

xt::xarray<float> EmotiEffLibRecognizerTorch::preprocess(const cv::Mat& img) {
    cv::Mat resized, float_img, normalized;

    // Resize the image to (img_size, img_size)
    cv::resize(img, resized, cv::Size(imgSize_, imgSize_));

    // Convert to float and scale to [0, 1]
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // Normalize using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    cv::merge(channels, normalized);

    // Convert HWC OpenCV Mat to CHW xtensor
    std::vector<float> chwData;
    chwData.reserve(3 * imgSize_ * imgSize_);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < imgSize_; ++h) {
            for (int w = 0; w < imgSize_; ++w) {
                chwData.push_back(normalized.at<cv::Vec3f>(h, w)[c]);
            }
        }
    }

    // Adapt vector to xt::xarray<float> with NCHW shape
    return xt::adapt(chwData, {1, 3, imgSize_, imgSize_});
}

void EmotiEffLibRecognizerTorch::configParser(const EmotiEffLibConfig& config) {
    if (!config.modelName.empty()) {
        modelName_ = config.modelName;
    }

    if (config.featureExtractorPath.empty()) {
        throw std::runtime_error(
            "featureExtractorPath MUST be specified in the EmotiEffLibConfig.");
    } else {
        torch::jit::script::Module model = torch::jit::load(config.featureExtractorPath);
        model.to(torch::Device(torch::kCPU));
        model.eval();
        models_.push_back(model);
        featureExtractorIdx_ = models_.size() - 1;
    }
    if (!config.classifierPath.empty()) {
        torch::jit::script::Module model = torch::jit::load(config.classifierPath);
        model.to(torch::Device(torch::kCPU));
        model.eval();
        models_.push_back(model);
        classifierIdx_ = models_.size() - 1;
    }
}
} // namespace EmotiEffLib
