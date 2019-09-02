cmake_minimum_required(VERSION 2.8)
project(ASLAM2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall   -O3 -march=native -pthread")

#------------------------------------------
#                    G2O
#------------------------------------------
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package( G2O REQUIRED )
include_directories(${G2O_INCLUDE_DIR})
find_package( Cholmod )
include_directories(${CHOLMOD_INCLUDE_DIR})
find_package(CSparse)
include_directories(${CSPARSE_INCLUDE_DIR})
list(APPEND G2O_LIBS
  g2o_cli g2o_ext_freeglut_minimal g2o_simulator
  g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
  g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
  g2o_types_sba g2o_types_slam3d g2o_csparse_extension
  g2o_opengl_helper g2o_solver_dense g2o_stuff
  g2o_types_sclam2d g2o_parser g2o_solver_pcg
  g2o_types_data g2o_types_sim3
)

#------------------------------------------
#                    OpenCV
#------------------------------------------
find_package( OpenCV REQUIRED )
include_directories(${OPENCV_INCLUDE_DIRS})

#------------------------------------------
#                    Eigen
#------------------------------------------
find_package(Eigen3 3.0 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

#------------------------------------------
#                    DBoW3
#------------------------------------------
find_package( DBoW3 REQUIRED )
include_directories( ${DBoW3_INCLUDE_DIRS} )


#------------------------------------------
#                    Pangolin
#------------------------------------------
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

#------------------------------------------
#                    PCL
#------------------------------------------
find_package(PCL REQUIRED COMPONENTS common registration segmentation)
add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})

#------------------------------------------
#                    fast
#------------------------------------------
FIND_PACKAGE(fast REQUIRED)
include_directories(${fast_INCLUDE_DIRS})

#------------------------------------------
#                    opengv
#------------------------------------------
find_package(opengv REQUIRED)

#------------------------------------------
#                    Octomap
#------------------------------------------
find_package(octomap)
include_directories(${OCTOMAP_INCLUDE_DIRS})

#------------------------------------------
#                    CPUTSDF
#------------------------------------------
#find_package(CPUTSDF)
#include_directories(${CPUTSDF_INCLUDE_DIR})

#------------------------------------------
#                    Sophus
#------------------------------------------
find_package(Sophus)
include_directories(${Sophus_INCLUDE_DIRS})

#find_package(vikit_common REQUIRED)
#include_directories(${vikit_common_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/src)

add_library(${PROJECT_NAME} SHARED
    # Core
    src/Core/KeyFrame.cpp
    src/Core/Map.cpp
    src/Core/Frame.cpp
    src/Core/Landmark.cpp
    src/Core/GraphNode.cpp
    src/Core/PinholeCamera.cpp
    src/Core/Feature.cpp

    # System
    src/System/System.cpp
    src/System/Tracking.cpp
    src/System/Mapping.cpp
    src/System/Converter.cpp
    src/System/Random.cpp
    src/System/LocalMap.cpp

    # Features
    src/Features/ORBextractor.cpp
    src/Features/Matcher.cpp
    src/Features/DetectorAdjuster.cpp
    src/Features/Extractor.cpp
    src/Features/StatefulFeatureDetector.cpp
    src/Features/SVOextractor.cpp
    src/Features/VideoDynamicAdaptedFeatureDetector.cpp
   src/Features/VideoGridAdaptedFeatureDetector.cpp

    # IO
    src/IO/Dataset.cpp
    src/IO/DatasetCORBS.cpp
    src/IO/DatasetICL.cpp
    src/IO/DatasetTUM.cpp
    src/IO/DatasetMicrosoft.cpp
    src/IO/DatasetAICL.cpp

    # Drawer
    src/Drawer/MapDrawer.cpp
    src/Drawer/Viewer.cpp
    src/Drawer/DenseMapDrawer.cpp

    # Solver
    src/Solver/PoseGraph.cpp
    src/Solver/Icp.cpp
    src/Solver/SE3solver.cpp
    src/Solver/PnPRansac.cpp
    src/Solver/BundleAdjustment.cpp
    src/Solver/Gicp.cpp

    # Place Recognition
    src/PlaceRecognition/KeyFrameDatabase.cpp
    src/PlaceRecognition/LoopClosing.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${DBoW3_LIBS}
${G2O_LIBS}
${CHOLMOD_LIBRARIES}
${CSPARSE_LIBRARY}
${PCL_LIBRARIES}
${OCTOMAP_LIBRARIES}
${fast_LIBRARIES}
opengv
#${CPUTSDF_LIBRARY}
${Sophus_LIBRARIES}
#${vikit_common_LIBRARIES}
)

add_executable(main main.cpp)
target_link_libraries(main ${PROJECT_NAME})
