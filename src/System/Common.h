#ifndef COMMON_H
#define COMMON_H

#include <DBoW3/Vocabulary.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <memory>
#include <opengv/types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>

#define SMART_POINTER_TYPEDEFS(T)         \
    typedef std::unique_ptr<T> UniquePtr; \
    typedef std::shared_ptr<T> Ptr;       \
    typedef std::shared_ptr<const T> ConstPtr

#define forn(i, n) for (size_t i = 0; i < (n); i++)
#define fornr(i, n) for (size_t i = (n)-1; 0 <= i; i--)
#define forsn(i, s, n) for (size_t i = (s); i < (n); i++)
#define forsnr(i, s, n) for (size_t i = (n)-1; (s) <= i; i--)
#define forall(it, X) for (decltype((X).begin()) it = (X).begin(); it != (X).end(); it++)
#define forallr(it, X) for (decltype((X).rbegin()) it = (X).rbegin(); it != (X).rend(); it++)

// Stream definitions
#define INFO_STREAM(x) std::cout << "[INFO] " << x << std::endl;
#define WARNING_STREAM(x) std::cout << "\033[33m[WARN] " << x << "\033[0m" << std::endl;
#define ERROR_STREAM(x) std::cout << "\033[31m[ERROR] " << x << "\033[0m" << std::endl;

// Common definitions
typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector4d Vec4;
typedef Eigen::Matrix3d Mat33;
typedef Eigen::Matrix4d Mat44;
typedef Eigen::VectorXd VecX;
typedef Eigen::Matrix<double, 6, 6> Mat66;
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix2d Mat22;
typedef Eigen::Quaterniond Quat;
typedef Sophus::SE3d SE3;

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Vec3)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Vec2)

typedef pcl::PointXYZRGB PointColor;
typedef pcl::PointCloud<PointColor> PointCloudColor;
typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef pcl::PointXYZRGBNormal PointColorNormal;
typedef pcl::PointCloud<PointColorNormal> PointCloudColorNormal;
typedef pcl::PointNormal PointNormal;
typedef pcl::PointCloud<PointNormal> PointCloudNormal;

class Frame;
typedef std::shared_ptr<Frame> FramePtr;

class KeyFrame;
typedef std::shared_ptr<KeyFrame> KeyFramePtr;

class Feature;
typedef std::shared_ptr<Feature> FeaturePtr;

class Landmark;
typedef std::shared_ptr<Landmark> LandmarkPtr;

class PinholeCamera;
typedef std::shared_ptr<PinholeCamera> CameraPtr;

class Map;
typedef std::shared_ptr<Map> MapPtr;

class MapDrawer;
typedef std::shared_ptr<MapDrawer> MapDrawerPtr;

class Extractor;
typedef std::shared_ptr<Extractor> ExtractorPtr;

class LocalMap;
typedef std::shared_ptr<LocalMap> LocalMapPtr;

class GraphNode;
typedef std::shared_ptr<GraphNode> NodePtr;

class Mapping;
typedef std::shared_ptr<Mapping> MappingPtr;

class Tracking;
typedef std::shared_ptr<Tracking> TrackingPtr;

class LoopClosing;
typedef std::shared_ptr<LoopClosing> LoopClosingPtr;

class Dataset;
typedef std::shared_ptr<Dataset> DatasetPtr;

typedef DBoW3::Vocabulary Vocabulary;
typedef std::shared_ptr<Vocabulary> VocabularyPtr;

class KeyFrameDatabase;
typedef std::shared_ptr<KeyFrameDatabase> KeyFrameDatabasePtr;

class Viewer;
typedef std::shared_ptr<Viewer> ViewerPtr;

class System;
typedef std::shared_ptr<System> SystemPtr;

class DenseMapDrawer;
typedef std::shared_ptr<DenseMapDrawer> DenseMapPtr;

#endif
