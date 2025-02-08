#include "emotiefflib/facial_analysis.h"

// TODO: include headers under ifdef torch
#include <torch/torch.h>

namespace EmotiEffLib {
EmotiEffLibRecognizerTorch::EmotiEffLibRecognizerTorch(const std::string& pathToModel,
                                                       const std::string& device) {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Hello World!! Here are my torch tensors::" << tensor << "!?" << std::endl;
}
} // namespace EmotiEffLib
