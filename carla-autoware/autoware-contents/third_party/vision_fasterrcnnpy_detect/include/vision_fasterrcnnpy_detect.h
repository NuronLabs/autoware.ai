#ifndef VISION_FASTERRCNNPY_DETECT_H_
#define VISION_FASTERRCNNPY_DETECT_H_

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "rect_class_score.h"

namespace SSD
{
  enum SSDDetectorClasses
  {
    BACKGROUND,
    PLANE, BICYCLE, BIRD, BOAT,
    BOTTLE, BUS, CAR, CAT, CHAIR,
    COW, TABLE, DOG, HORSE,
    MOTORBIKE, PERSON, PLANT,
    SHEEP, SOFA, TRAIN, TV, NUM_CLASSES
  };
}

class PytorchDetector
{
public:
  PytorchDetector(const std::string& in_pre_trained_model_file, const cv::Scalar& in_mean_value, bool in_use_gpu, unsigned int in_gpu_id);

  std::vector <  RectClassScore<float>  > Detect(const cv::Mat& img);

private:
  void SetMean(const cv::Scalar &in_mean_value);

  void WrapInputLayer(std::vector<cv::Mat> *input_channels);

  void Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels);

  torch::jit::script::Module LoadModel(const std::string& path);

  // The return is defined as an auto because the dict output returned by
  // Torchscript does not have an easily identifiable type.
  auto RunInference(const cv::Mat& img);

private:
  std::shared_ptr <torch::jit::script::Module > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Scalar mean_;
};

#endif //VISION_FASTERRCNNPY_DETECT_H_
