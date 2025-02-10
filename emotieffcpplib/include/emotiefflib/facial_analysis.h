#ifndef FACIAL_ANALYSIS_H
#define FACIAL_ANALYSIS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <xtensor/xarray.hpp>

namespace EmotiEffLib {

std::vector<std::string> getAvailableBackends();

class EmotiEffLibRecognizer {
public:
    virtual ~EmotiEffLibRecognizer() = default;
    static std::unique_ptr<EmotiEffLibRecognizer> createInstance(const std::string& engine,
                                                                 const std::string& modelPath);
    static std::unique_ptr<EmotiEffLibRecognizer> createInstance(const std::string& engine,
                                                                 const std::string& dirWithModels,
                                                                 const std::string& modelName);

protected:
    EmotiEffLibRecognizer(const std::string& modelPath);
    virtual xt::xarray<float> preprocess(const cv::Mat& img) = 0;
    // virtual void extractFeatures(const cv::Mat& img) = 0;
    //// def classify_emotions(features: np.ndarray, logits: bool = True) -> Tuple[List[str],
    // np.ndarray]:
    //// def classify_engagement(features: np.ndarray, sliding_window_width: int = 128):
    //// def predict_emotions(face_img: Union[np.ndarray, List[np.ndarray]], logits: bool = True)
    //-> Tuple[List[str], np.ndarray]:
    //// def predict_engagement(features: np.ndarray, sliding_window_width: int = 128):
protected:
    std::vector<std::string> idxToEngagementClass_ = {"Distracted", "Engaged"};
    std::vector<std::string> idxToEmotionClass_;
    std::string modelName_;
    bool isMtl_;
    int imgSize_;
};

} // namespace EmotiEffLib

#endif
