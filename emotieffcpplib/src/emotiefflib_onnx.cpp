#include "emotiefflib/facial_analysis.h"

#include <filesystem>

#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

namespace EmotiEffLib {
EmotiEffLibRecognizerOnnx::EmotiEffLibRecognizerOnnx(const std::string& pathToModel) {
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

    // std::cout << "Model loaded successfully!" << std::endl;
}
} // namespace EmotiEffLib
