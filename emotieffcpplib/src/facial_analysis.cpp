#include "emotiefflib/facial_analysis.h"

#include <filesystem>

// TODO: include headers under ifdef torch
#include <torch/script.h>
#include <torch/torch.h>

namespace fs = std::filesystem;

namespace EmotiEffLib {
EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& pathToModel) {
    std::string model_name = fs::path(pathToModel).filename().string();
    bool is_mtl = model_name.find("_mtl") != std::string::npos;
    bool is_b0 = model_name.find("_b0_") != std::string::npos;
    int img_size = is_b0 ? 224 : 260;
    std::cout << pathToModel << std::endl;
    auto model = torch::jit::load(pathToModel);
}
} // namespace EmotiEffLib
