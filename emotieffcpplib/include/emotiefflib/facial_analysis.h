#ifndef FACIAL_ANALYSIS_H
#define FACIAL_ANALYSIS_H

#include <string>

namespace EmotiEffLib {
class EmotiEffLibRecognizerTorch {
public:
    EmotiEffLibRecognizerTorch(const std::string& pathToModel, const std::string& device);

private:
    int imgSize_;
};
} // namespace EmotiEffLib

#endif
