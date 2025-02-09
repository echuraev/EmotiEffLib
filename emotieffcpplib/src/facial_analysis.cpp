#include "emotiefflib/facial_analysis.h"

#include <filesystem>

// TODO: include headers under ifdef torch
#include <torch/script.h>
#include <torch/torch.h>

namespace fs = std::filesystem;

namespace EmotiEffLib {
EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& pathToModel) {
    torch::jit::script::Module model = torch::jit::load(pathToModel);
    std::string model_name = fs::path(pathToModel).filename().string();
    isMtl_ = model_name.find("_mtl") != std::string::npos;
    bool isB0 = model_name.find("_b0_") != std::string::npos;
    bool is7 = model_name.find("_7") != std::string::npos;
    imgSize_ = isB0 ? 224 : 260;
    if (is7) {
        idxToEmotionClass_.resize(7);
        idxToEmotionClass_[0] = "Anger";
        idxToEmotionClass_[1] = "Disgust";
        idxToEmotionClass_[2] = "Fear";
        idxToEmotionClass_[3] = "Happiness";
        idxToEmotionClass_[4] = "Neutral";
        idxToEmotionClass_[5] = "Sadness";
        idxToEmotionClass_[6] = "Surprise";
    } else {
        idxToEmotionClass_.resize(8);
        idxToEmotionClass_[0] = "Anger";
        idxToEmotionClass_[1] = "Contempt";
        idxToEmotionClass_[2] = "Disgust";
        idxToEmotionClass_[3] = "Fear";
        idxToEmotionClass_[4] = "Happiness";
        idxToEmotionClass_[5] = "Neutral";
        idxToEmotionClass_[6] = "Sadness";
        idxToEmotionClass_[7] = "Surprise";
    }
    auto x = torch::randn({1, 3, 224, 224});
    auto output = model.forward({x}).toTensor();
    std::cout << "Output shape: " << output.sizes() << std::endl;
}
} // namespace EmotiEffLib
