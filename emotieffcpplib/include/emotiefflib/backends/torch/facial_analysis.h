#ifndef BACKENDS_TORCH_FACIAL_ANALYSIS_H
#define BACKENDS_TORCH_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

namespace EmotiEffLib {
class EmotiEffLibRecognizerTorch : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerTorch(const std::string& modelPath);
};
} // namespace EmotiEffLib

#endif
