#ifndef BACKENDS_TORCH_FACIAL_ANALYSIS_H
#define BACKENDS_TORCH_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

namespace EmotiEffLib {
class EmotiEffLibRecognizerTorch : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerTorch(const std::string& modelPath);
    EmotiEffLibRecognizerTorch(const std::string& dirWithModels, const std::string& modelName);
    cv::Mat preprocess(const cv::Mat& img) override;
};
} // namespace EmotiEffLib

#endif
