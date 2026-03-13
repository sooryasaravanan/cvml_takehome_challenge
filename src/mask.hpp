#pragma once

#include <stdexcept>
#include <string>
#include "opencv2/opencv.hpp"
#include "view.hpp"

// function: given path to a folder of masks, and a const ref to a view, load and return the mask for the image.
// scale the mask to the same size as the image, as indicated by the view's intrinsic.
// return the mask as a cv::Mat.
cv::Mat loadMask(const std::string &mask_folder, const View &view) {
  // load the mask from the folder:
  std::string mask_path = mask_folder + "/" + view.filename;

  // change the extension of the mask path to .png:
  mask_path = mask_path.substr(0, mask_path.find_last_of('.')) + ".png";

  cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
  
  // Check if mask was loaded successfully
  if (mask.empty()) {
    throw std::runtime_error("Failed to load mask from: " + mask_path);
  }
  
  // scale the mask to the same size as the image, as indicated by the view's intrinsic:
  cv::Mat scaled_mask;
  cv::resize(mask, scaled_mask, cv::Size(view.intrinsic.cols, view.intrinsic.rows));
  return scaled_mask;
}
