#ifndef BACKENDS_ONNX_FACIAL_ANALYSIS_H
#define BACKENDS_ONNX_FACIAL_ANALYSIS_H

#include "emotiefflib/facial_analysis.h"

namespace EmotiEffLib {
class EmotiEffLibRecognizerOnnx : public EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizerOnnx(const std::string& modelPath);
};
} // namespace EmotiEffLib

#endif
