#include "emotiefflib/backends/onnx/facial_analysis.h"

#include <filesystem>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace fs = std::filesystem;

namespace {
Ort::Value xarray2tensor(const xt::xarray<float>& xarray) {
    auto xtensor = xt::eval(xarray);
    // Extract shape
    std::vector<int64_t> shape(xtensor.shape().begin(), xtensor.shape().end());

    // Create new buffer
    std::vector<float> buffer(xtensor.begin(), xtensor.end());

    // Create ONNX Runtime memory info (CPU)
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Create ONNX Runtime tensor
    Ort::Value onnx_tensor = Ort::Value::CreateTensor<float>(
        memory_info, buffer.data(), buffer.size(), shape.data(), shape.size());

    // Verify tensor is valid
    if (!onnx_tensor.IsTensor()) {
        throw std::runtime_error("Error during ONNX tensor creation!");
    }
    return onnx_tensor;
}

xt::xarray<float> tensor2xarray(const Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        throw std::runtime_error("Input ONNX Value is not a tensor!");
    }

    // Get shape information
    Ort::TensorTypeAndShapeInfo tensor_info = tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();

    // Get raw data pointer (use GetTensorData instead of GetTensorMutableData)
    const float* data_ptr = tensor.GetTensorData<float>();
    size_t element_count = tensor_info.GetElementCount();

    // Convert shape to xtensor format (size_t instead of int64_t)
    std::vector<size_t> xtensor_shape(shape.begin(), shape.end());

    // Create an xt::xarray and copy the data
    xt::xarray<float> result = xt::zeros<float>(xtensor_shape);
    std::copy(data_ptr, data_ptr + element_count, result.begin());
    return result;
}
} // namespace

namespace EmotiEffLib {
EmotiEffLibRecognizerOnnx::EmotiEffLibRecognizerOnnx(const std::string& modelPath)
    : EmotiEffLibRecognizer(modelPath) {
    // auto providers = Ort::GetAvailableProviders();
    // for (auto provider : providers) {
    //     std::cout << provider << std::endl;
    // }
    //  Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

    // Define session options
    Ort::SessionOptions session_options;

    // Load the ONNX model
    Ort::Session model(env, modelPath.c_str(), session_options);
    models_.push_back(std::move(model));

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
    : EmotiEffLibRecognizer(modelName) {
    fs::path featureExtractorPath(dirWithModels);
    featureExtractorPath /= modelName + ".pt";
    fs::path classifierPath(dirWithModels);
    classifierPath /= "classifier_" + modelName + ".pt";
}

EmotiEffLibRes EmotiEffLibRecognizerOnnx::precictEmotions(const cv::Mat& faceImg, bool logits) {
    auto imgTensor = preprocess(faceImg);

    auto& session = models_[0];

    Ort::AllocatorWithDefaultOptions allocator;
    // TODO:
    //// Check numInputNodes
    // size_t numInputNodes = session.GetInputCount();
    //// Check numOutputNodes
    // size_t numOutputNodes = session.GetOutputCount();

    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(xarray2tensor(imgTensor));

    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> output_names = {output_name.get()};

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                      input_tensors.data(), 1, output_names.data(), 1);

    // Print first few output values
    auto scores = tensor2xarray(output_tensors[0]);

    // std::vector<std::string> labels = {"None"};
    // return {labels, scores};
    return processScores(scores, logits);
}

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
