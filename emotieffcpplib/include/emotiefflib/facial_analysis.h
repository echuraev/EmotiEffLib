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

struct EmotiEffLibConfig {
    std::string backend;
    std::string featureExtractorPath;
    std::string classifierPath;
    std::string modelName = "";
};

class EmotiEffLibRecognizer {
public:
    virtual ~EmotiEffLibRecognizer() = default;
    static std::unique_ptr<EmotiEffLibRecognizer>
    createInstance(const std::string& backend, const std::string& fullPipelineModelPath);
    static std::unique_ptr<EmotiEffLibRecognizer> createInstance(const EmotiEffLibConfig& config);
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
    virtual void initRecognizer(const std::string& modelPath);
    EmotiEffLibRes processScores(const xt::xarray<float>& scores, bool logits);
    virtual void configParser(const EmotiEffLibConfig& config) = 0;
    virtual xt::xarray<float> preprocess(const cv::Mat& img) = 0;

private:
    static void checkBackend(const std::string& backend);

protected:
    std::string modelName_ = "";
    int fullPipelineModelIdx_ = -1;
    int featureExtractorIdx_ = -1;
    int classifierIdx_ = -1;
    std::vector<std::string> idxToEngagementClass_ = {"Distracted", "Engaged"};
    std::vector<std::string> idxToEmotionClass_;
    bool isMtl_;
    int imgSize_;
};

} // namespace EmotiEffLib

#endif
