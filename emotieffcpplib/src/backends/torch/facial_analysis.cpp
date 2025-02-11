#include "emotiefflib/backends/torch/facial_analysis.h"

#include <filesystem>

#include <xtensor/xadapt.hpp>

namespace fs = std::filesystem;

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
EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& modelPath)
    : EmotiEffLibRecognizer(modelPath) {
    torch::jit::script::Module model = torch::jit::load(modelPath);
    models.push_back(model);
    bool isB0 = modelName_.find("_b0_") != std::string::npos;
    imgSize_ = isB0 ? 224 : 260;
}

EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& dirWithModels,
                                                       const std::string& modelName)
    : EmotiEffLibRecognizer(modelName) {
    fs::path featureExtractorPath(dirWithModels);
    featureExtractorPath /= modelName + ".pt";
    fs::path classifierPath(dirWithModels);
    classifierPath /= "classifier_" + modelName + ".pt";
    torch::jit::script::Module featureExtractor = torch::jit::load(featureExtractorPath);
    torch::jit::script::Module classifier = torch::jit::load(classifierPath);
    models.push_back(featureExtractor);
    models.push_back(classifier);
    bool isB0 = modelName_.find("_b0_") != std::string::npos;
    imgSize_ = isB0 ? 224 : 260;
}

EmotiEffLibRes EmotiEffLibRecognizerTorch::precictEmotions(const cv::Mat& faceImg, bool logits) {
    auto imgTensor = preprocess(faceImg);
    auto input = xarray2tensor(imgTensor);
    // std::cout << "Tensor shape: (";
    // for (size_t i = 0; i < input.sizes().size(); ++i) {
    //     std::cout << input.sizes()[i];
    //     if (i < input.sizes().size() - 1) {
    //         std::cout << ", ";
    //     }
    // }
    // std::cout << ")" << std::endl;
    auto scoresTensor = models[0].forward({input}).toTensor();
    auto scores = tensor2xarray(scoresTensor);

    return processScores(scores, logits);
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
    //// Convert HWC OpenCV Mat to xtensor
    // std::vector<float> hwcData;
    // hwcData.assign((float*)normalized.datastart, (float*)normalized.dataend);

    //// to NHWC
    // return xt::adapt(hwcData, {1, imgSize_, imgSize_, 3});
}
} // namespace EmotiEffLib
