#ifndef FACIAL_ANALYSIS_H
#define FACIAL_ANALYSIS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <xtensor/xarray.hpp>

namespace EmotiEffLib {

std::vector<std::string> getAvailableBackends();
std::vector<std::string> getSupportedModels();

struct EmotiEffLibRes {
    std::vector<std::string> labels;
    xt::xarray<float> scores;
};

class EmotiEffLibRecognizer {
public:
    virtual ~EmotiEffLibRecognizer() = default;
    static std::unique_ptr<EmotiEffLibRecognizer> createInstance(const std::string& engine,
                                                                 const std::string& modelPath);
    static std::unique_ptr<EmotiEffLibRecognizer> createInstance(const std::string& engine,
                                                                 const std::string& dirWithModels,
                                                                 const std::string& modelName);
    // virtual xt::xarray<float> extractFeatures(const xt::xarray<float>& faceImg) = 0;
    // virtual xt::xarray<float> extractFeatures(const std::vector<xt::xarray<float>>& faceImgs) =
    // 0; virtual EmotiEffLibRes classifyEmotions(const xt::xarray<float>& features,
    //                                         bool logits = true) = 0;
    // virtual EmotiEffLibRes classifyEngagement(const xt::xarray<float>& features,
    //                                           int slidingWindowWidth = 128) = 0;
    virtual EmotiEffLibRes precictEmotions(const cv::Mat& faceImg, bool logits = true) = 0;
    // virtual EmotiEffLibRes precictEmotions(const std::vector<cv::Mat>& faceImgs,
    //                                        bool logits = true) = 0;
    // virtual EmotiEffLibRes precictEngagement(const std::vector<cv::Mat>& faceImgs,
    //                                          int slidingWindowWidth = 128) = 0;

protected:
    EmotiEffLibRecognizer(const std::string& modelPath);
    virtual xt::xarray<float> preprocess(const cv::Mat& img) = 0;
    EmotiEffLibRes processScores(const xt::xarray<float>& scores, bool logits);

protected:
    std::vector<std::string> idxToEngagementClass_ = {"Distracted", "Engaged"};
    std::vector<std::string> idxToEmotionClass_;
    std::string modelName_;
    bool isMtl_;
    int imgSize_;
};

} // namespace EmotiEffLib

#endif
