#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include "nlohmann/json.hpp"

// intrinsic parameters.
// this is a pinhole camera model, with zero distortion or skew.
struct Intrinsic {
  Eigen::Matrix3f K; // intrinsic matrix K
  int rows; // height of the image
  int cols; // width of the image
};

// extrinsic parameters.
struct Extrinsic {
  Eigen::Matrix3f R; // rotation matrix
  Eigen::Vector3f c; // camera center
};

// 3D ray representation: origin point and direction vector
struct Ray {
  Eigen::Vector3f origin;    // ray origin point
  Eigen::Vector3f direction; // ray direction vector (normalized)
};

struct View {
  Intrinsic intrinsic;
  Extrinsic extrinsic;
  std::string filename;
  
  // Calculate the 3D ray corresponding to a given pixel (u, v)
  // u is the column (x-coordinate), v is the row (y-coordinate)
  Ray pixelToRay(float u, float v) const {
    Ray ray;
    
    // Ray origin is the camera center in world coordinates
    ray.origin = extrinsic.c;
    
    // Convert pixel coordinates to normalized camera coordinates
    // [x, y, 1]^T = K^(-1) * [u, v, 1]^T
    Eigen::Vector3f pixel_homogeneous(u, v, 1.0f);
    Eigen::Vector3f cam_coords = intrinsic.K.inverse() * pixel_homogeneous;
    
    // Direction in camera space is [x, y, 1] normalized
    Eigen::Vector3f dir_cam(cam_coords(0), cam_coords(1), 1.0f);
    dir_cam.normalize();
    
    // Transform direction from camera space to world space
    // R transforms world->camera, so R^T transforms camera->world
    ray.direction = extrinsic.R.transpose() * dir_cam;
    
    return ray;
  }
};

// utility function to load views from a JSON file.
std::vector<View> loadViewsFromJson(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  
  // Check if file is empty
  file.seekg(0, std::ios::end);
  if (file.tellg() == 0) {
    throw std::runtime_error("File is empty: " + filename);
  }
  file.seekg(0, std::ios::beg);
  
  nlohmann::json json;
  try {
    file >> json;
  } catch (const nlohmann::json::parse_error& e) {
    throw std::runtime_error("JSON parse error in " + filename + ": " + e.what());
  }
  
  if (json.is_null() || !json.contains("views")) {
    throw std::runtime_error("Invalid JSON structure: missing 'views' key in " + filename);
  }
  
  std::vector<View> views;
  
  // Parse the "views" array from JSON
  for (const auto& view_json : json["views"]) {
    View view;
    
    // Parse intrinsic parameters
    const auto& intrinsic_json = view_json["intrinsic"];
    double focal_length = intrinsic_json["focal_length"];
    const auto& principal_point = intrinsic_json["principal_point"];
    double cx = principal_point[0];
    double cy = principal_point[1];
    int width = intrinsic_json["width"];
    int height = intrinsic_json["height"];
    
    // Build intrinsic matrix K
    // K = [f  0  cx]
    //     [0  f  cy]
    //     [0  0  1 ]
    view.intrinsic.K << focal_length, 0.0f, static_cast<float>(cx),
                        0.0f, focal_length, static_cast<float>(cy),
                        0.0f, 0.0f, 1.0f;
    view.intrinsic.rows = height;
    view.intrinsic.cols = width;
    
    // Parse extrinsic parameters
    const auto& extrinsic_json = view_json["extrinsic"];
    const auto& rotation = extrinsic_json["rotation"];
    const auto& center = extrinsic_json["center"];
    
    // Build rotation matrix R (3x3)
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        view.extrinsic.R(i, j) = static_cast<float>(rotation[i][j]);
      }
    }
    
    // Build camera center vector c (3x1)
    for (int i = 0; i < 3; ++i) {
      view.extrinsic.c(i) = static_cast<float>(center[i]);
    }
    
    // Parse filename
    view.filename = view_json["filename"];
    
    views.push_back(view);
  }
  
  return views;
}
