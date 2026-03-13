#include <iostream>

#include "Eigen/Dense"
#include "nlohmann/json.hpp"
#include "opencv2/opencv.hpp"

#include "ply_stream.hpp"
#include "view.hpp"
#include "mask.hpp"


// this defined the ground-plane for this dataset.
const Eigen::Vector3f ground_plane_point = Eigen::Vector3f(-112.66657257080078, -117.26902770996094, 29.7415714263916);
const Eigen::Vector3f ground_plane_normal = Eigen::Vector3f(-0.009972016332064267, 0.001650051087892894, -0.9999489168060939);

int main() {

  // load views from json file:
  std::vector<View> views = loadViewsFromJson(std::string(PROJECT_ROOT) + "/data/views.json");
  std::cout << "Loaded " << views.size() << " views" << std::endl;

  // example: print the first 3 views:
  for (int i = 0; i < 3; i++) {
    std::cout << "--------------------------------" << std::endl;
    std::cout << "View " << i << ": " << views[i].filename << std::endl;
    std::cout << "Intrinsic: " << views[i].intrinsic.K << std::endl;
    std::cout << "Image size: " << views[i].intrinsic.rows << "x" << views[i].intrinsic.cols << std::endl;
    std::cout << "Extrinsic: " << views[i].extrinsic.R << std::endl;
    std::cout << "Center: " << views[i].extrinsic.c.transpose() << std::endl;
    std::cout << "--------------------------------" << std::endl;
  }

  // example: load and print the mask for the first view:
  cv::Mat mask = loadMask(std::string(PROJECT_ROOT) + "/data/masks", views[0]);
  std::cout << "--------------------------------" << std::endl;
  std::cout << "loaded mask for view " << views[0].filename << " with size " << mask.size() << std::endl;
  std::cout << "number of non-zero pixels in mask: " << cv::countNonZero(mask) << std::endl;
  std::cout << "--------------------------------" << std::endl;

  // example: extract the boundary of the mask for the first view:
  MultiPolygon2D boundary = extractBoundary(mask);
  std::cout << "--------------------------------" << std::endl;
  std::cout << "Boundary of mask for view " << views[0].filename << ": " << boundary.size() << " polygons" << std::endl;
  // std::cout << bg::wkt(boundary) << std::endl;
  std::cout << "--------------------------------" << std::endl;

  // example: get the 3D ray for the central pixel in the first view:
  Ray ray = views[0].pixelToRay(views[0].intrinsic.cols / 2, views[0].intrinsic.rows / 2);
  std::cout << "--------------------------------" << std::endl;
  std::cout << "3D ray for view " << views[0].filename << " at central pixel: " << std::endl;
  std::cout << "3D ray for central pixel: " << ray.direction.transpose() << std::endl;
  std::cout << "3D ray origin: " << ray.origin.transpose() << std::endl;
  std::cout << "--------------------------------" << std::endl;

  // // example: write a few points and line segments to a ply file:
  // PointPlyStream ply_stream(std::string(PROJECT_ROOT) + "/output/pt_cloud.ply");
  // ply_stream.WriteHeader({"float x", "float y", "float z", "uchar red", "uchar green", "uchar blue"});
  // ply_stream << 0.f << 0.f << 0.f << (unsigned char)255 << (unsigned char)0 << (unsigned char)0;
  // ply_stream << 100.f << 0.f << 0.f << (unsigned char)255 << (unsigned char)0 << (unsigned char)0;
  // ply_stream << 100.f << 100.f << 0.f << (unsigned char)255 << (unsigned char)0 << (unsigned char)0;
  // ply_stream << 0.f << 100.f << 0.f << (unsigned char)255 << (unsigned char)0 << (unsigned char)0;
  // generate3DLineSegment({5, 5, 0}, Eigen::Vector3f(95, 95, 0), 1.f, 0, 255, 0, ply_stream);
  // generate3DLineSegment({50, 50, 0}, Eigen::Vector3f(50,50,100), 1.f, 0, 0, 255, ply_stream);  
  
  

  // TODO: do your stuff here



  return 0;
}
