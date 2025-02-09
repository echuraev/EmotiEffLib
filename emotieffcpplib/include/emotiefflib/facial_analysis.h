#ifndef FACIAL_ANALYSIS_H
#define FACIAL_ANALYSIS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace EmotiEffLib {

class EmotiEffLibRecognizer {
public:
    EmotiEffLibRecognizer(const std::string& modelPath);
    virtual ~EmotiEffLibRecognizer() = default;

protected:
    // virtual cv::Mat preprocess(const cv::Mat& img) = 0;
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

std::vector<std::string> getAvailableBackends();
std::unique_ptr<EmotiEffLibRecognizer> createEmotiEffLibRecognizer(const std::string& engine,
                                                                   const std::string& modelPath);

} // namespace EmotiEffLib

#endif
