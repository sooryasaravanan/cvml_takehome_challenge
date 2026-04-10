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
  
  

  std::string data_root = std::string(PROJECT_ROOT) + "/data";
  std::string output_root = std::string(PROJECT_ROOT) + "/output";

  MultiPolygon2D combined;
  int morph_size = 5;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_size, morph_size));
  float plane_d = ground_plane_normal.dot(ground_plane_point);

  for (size_t i = 0; i < views.size(); i++) {
    cv::Mat m = loadMask(data_root + "/masks", views[i]);

    cv::morphologyEx(m, m, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(m, m, cv::MORPH_OPEN, kernel);

    MultiPolygon2D bnd = extractBoundary(m);
    if (bnd.empty()) continue;

    // filter out tiny contours
    MultiPolygon2D filtered;
    for (const auto &p : bnd) {
      if (bg::area(p) > 500.0) filtered.push_back(p);
    }
    if (filtered.empty()) continue;

    // backproject each polygon onto the ground plane
    MultiPolygon2D world_poly;
    for (const auto &poly : filtered) {
      Polygon2D wp;
      for (const auto &pt : poly.outer()) {
        float u = bg::get<0>(pt);
        float v = bg::get<1>(pt);
        Ray r = views[i].pixelToRay(u, v);
        float denom = r.direction.dot(ground_plane_normal);
        float t = (ground_plane_point - r.origin).dot(ground_plane_normal) / denom;
        Eigen::Vector3f hit = r.origin + t * r.direction;
        bg::append(wp.outer(), Point2D(hit.x(), hit.y()));
      }
      if (wp.outer().size() >= 3) {
        auto &ring = wp.outer();
        if (bg::get<0>(ring.front()) != bg::get<0>(ring.back()) ||
            bg::get<1>(ring.front()) != bg::get<1>(ring.back())) {
          ring.push_back(ring.front());
        }
        bg::correct(wp);
        world_poly.push_back(wp);
      }
    }
    if (world_poly.empty()) continue;

    if (combined.empty()) {
      combined = world_poly;
    } else {
      MultiPolygon2D result;
      try {
        bg::union_(combined, world_poly, result);
        combined = result;
      } catch (...) {
        std::cerr << "Warning: union failed for view " << i << ", skipping" << std::endl;
      }
    }
    std::cout << "Processed view " << i + 1 << "/" << views.size() << ": " << views[i].filename << std::endl;
  }

  std::cout << "Combined mask has " << combined.size() << " polygons" << std::endl;

  // write combined mask boundary as green line segments to ply
  {
    PointPlyStream ply(output_root + "/combined_mask.ply");
    ply.WriteHeader({"float x", "float y", "float z", "uchar red", "uchar green", "uchar blue"});

    for (const auto &poly : combined) {
      const auto &ring = poly.outer();
      for (size_t j = 0; j + 1 < ring.size(); j++) {
        float x0 = bg::get<0>(ring[j]), y0 = bg::get<1>(ring[j]);
        float x1 = bg::get<0>(ring[j + 1]), y1 = bg::get<1>(ring[j + 1]);
        float z0 = (plane_d - ground_plane_normal.x() * x0 - ground_plane_normal.y() * y0) / ground_plane_normal.z();
        float z1 = (plane_d - ground_plane_normal.x() * x1 - ground_plane_normal.y() * y1) / ground_plane_normal.z();
        generate3DLineSegment(Eigen::Vector3f(x0, y0, z0), Eigen::Vector3f(x1, y1, z1),
                              0.1f, 0, 255, 0, ply);
      }
    }
    std::cout << "Wrote combined mask to " << output_root << "/combined_mask.ply" << std::endl;
  }

  return 0;
}
