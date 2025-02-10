#include "emotiefflib/backends/onnx/facial_analysis.h"
#include "emotiefflib/facial_analysis.h"

#include <onnxruntime_cxx_api.h>

#include <xtensor/xadapt.hpp>

namespace EmotiEffLib {
EmotiEffLibRecognizerOnnx::EmotiEffLibRecognizerOnnx(const std::string& modelPath)
    : EmotiEffLibRecognizer(modelPath) {
    auto providers = Ort::GetAvailableProviders();
    for (auto provider : providers) {
        std::cout << provider << std::endl;
    }
    // onnxruntime::InferenceSession session(onnxruntime::SessionOptions());
    // onnxruntime::Status status = session.Load(model_path);
    // if (!status.ok()) {
    //     std::cerr << "Error loading model: " << status.ErrorMessage() << std::endl;
    //     return -1;
    // }

    std::cout << "Model loaded successfully!" << std::endl;

    mean_ = {0.485, 0.456, 0.406};
    std_ = {0.229, 0.224, 0.225};
    if (modelName_.find("mbf_") != std::string::npos) {
        imgSize_ = 112;
        mean_ = {0.5, 0.5, 0.5};
        std_ = {0.5, 0.5, 0.5};
    } else if (modelName_.find("_b2_") != std::string::npos) {
        imgSize_ = 260;
    } else if (modelName_.find("ddamfnet") != std::string::npos) {
        imgSize_ = 112;
    } else {
        imgSize_ = 224;
    }
}

EmotiEffLibRecognizerOnnx::EmotiEffLibRecognizerOnnx(const std::string& dirWithModels,
                                                     const std::string& modelName)
    : EmotiEffLibRecognizer(modelName) {}

xt::xarray<float> EmotiEffLibRecognizerOnnx::preprocess(const cv::Mat& img) {
    cv::Mat resized_img, float_img, normalized_img;

    // Resize the image
    cv::resize(img, resized_img, cv::Size(imgSize_, imgSize_));

    // Convert to float32 and scale to [0, 1]
    resized_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // Normalize each channel
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }

    // Merge back the channels
    cv::merge(channels, normalized_img);

    // Convert HWC OpenCV Mat to CHW xtensor
    std::vector<float> chwData;
    chwData.reserve(3 * imgSize_ * imgSize_);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < imgSize_; ++h) {
            for (int w = 0; w < imgSize_; ++w) {
                chwData.push_back(normalized_img.at<cv::Vec3f>(h, w)[c]);
            }
        }
    }

    // Adapt vector to xt::xarray<float> with NCHW shape
    return xt::adapt(chwData, {1, 3, imgSize_, imgSize_});
}
} // namespace EmotiEffLib
