#include "emotiefflib/backends/onnx/facial_analysis.h"
#include "emotiefflib/facial_analysis.h"

#include <onnxruntime_cxx_api.h>

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

    // std::cout << "Model loaded successfully!" << std::endl;
}
} // namespace EmotiEffLib
