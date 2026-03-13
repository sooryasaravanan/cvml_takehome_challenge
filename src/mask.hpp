#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "view.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>

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

// function: given a mask as a cv::Mat, extract its boundary and return it as a boost::geometry multi_polygon.
// Define Boost Geometry types
namespace bg = boost::geometry;
using Point2D = bg::model::point<float, 2, bg::cs::cartesian>;
using Polygon2D = bg::model::polygon<Point2D>;
using MultiPolygon2D = bg::model::multi_polygon<Polygon2D>;

MultiPolygon2D extractBoundary(const cv::Mat &mask) {
  // Ensure mask is binary (0 or 255)
  cv::Mat binary_mask;
  if (mask.channels() == 1) {
    // Check if mask is already binary or needs thresholding
    // If mask has values > 0, threshold it. Otherwise, it might already be binary.
    double min_val, max_val;
    cv::minMaxLoc(mask, &min_val, &max_val);
    
    if (max_val > 1.0) {
      // Mask has non-binary values, threshold it
      // Use a low threshold (1) to preserve any non-zero pixels
      cv::threshold(mask, binary_mask, 1, 255, cv::THRESH_BINARY);
    } else {
      // Mask is already binary (0 or 1), scale to 0 or 255
      binary_mask = mask * 255;
    }
  } else {
    throw std::runtime_error("Mask must be a single-channel image");
  }
  
  // Find contours - RETR_EXTERNAL gets only external contours
  // CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binary_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  
  MultiPolygon2D multi_poly;
  
  // Convert each contour to a Boost Geometry polygon
  for (const auto &contour : contours) {
    if (contour.size() < 3) {
      // Skip contours with less than 3 points (can't form a polygon)
      continue;
    }
    
    Polygon2D poly;
    
    // Add points to the outer ring of the polygon
    for (const auto &pt : contour) {
      bg::append(poly.outer(), Point2D(static_cast<float>(pt.x), static_cast<float>(pt.y)));
    }
    
    // OpenCV findContours returns closed contours, but Boost Geometry
    // requires the first and last point to be the same for a closed ring
    // Close the polygon if needed
    if (poly.outer().size() > 0) {
      const auto &first_pt = poly.outer()[0];
      const auto &last_pt = poly.outer().back();
      // If not closed, close it
      if (bg::get<0>(first_pt) != bg::get<0>(last_pt) || 
          bg::get<1>(first_pt) != bg::get<1>(last_pt)) {
        bg::append(poly.outer(), first_pt);
      }
    }
    
    // Ensure polygon is closed and has minimum points
    if (poly.outer().size() < 4) {
      continue;  // Need at least 4 points for a closed polygon (including duplicate first/last)
    }
    
    // Correct the polygon (fixes orientation, removes self-intersections)
    bg::correct(poly);
    
    // Only add non-empty polygons
    if (poly.outer().size() >= 4) {
      multi_poly.push_back(poly);
    }
  }
  
  return multi_poly;
}

