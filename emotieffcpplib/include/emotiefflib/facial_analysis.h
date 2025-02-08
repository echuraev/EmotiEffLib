#ifndef FACIAL_ANALYSIS_H
#define FACIAL_ANALYSIS_H

#include <string>

class EmotiEffLibRecognizerTorch {
public:
    EmotiEffLibRecognizerTorch(const std::string& pathToModel, const std::string& device);

private:
    int imgSize_;
};

#endif
