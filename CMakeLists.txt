cmake_minimum_required(VERSION 3.8)
project(sfm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
#find_package(PCL REQUIRED)

include_directories(/usr/include/g2o)
set(G2O_LIBS
    g2o_core
    g2o_stuff
    g2o_types_slam3d
    g2o_types_slam2d
    g2o_solver_csparse
    cxsparse
)

include_directories(
    include
    /usr/local/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${g2o_INCLUDE_DIRS}
    #${PCL_INCLUDE_DIRS}
)

add_executable(dense_mapping_node
    src/dense_mapping_node.cpp
)

target_link_libraries(dense_mapping_node
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Sophus::Sophus
    ${G2O_LIBS}
    DBoW2
    #${PCL_LIBRARIES}
)

ament_target_dependencies(dense_mapping_node
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    tf2
    tf2_ros
    visualization_msgs
)

install(TARGETS dense_mapping_node DESTINATION lib/${PROJECT_NAME})

add_executable(sfm_node
    src/sfm_node.cpp
)

target_link_libraries(sfm_node
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Sophus::Sophus
    ${G2O_LIBS}
    DBoW2
    #${PCL_LIBRARIES}
)

ament_target_dependencies(sfm_node
    rclcpp
    sensor_msgs
    std_msgs
    geometry_msgs
    tf2
    tf2_ros
    visualization_msgs
)

target_compile_definitions(sfm_node PRIVATE DATASET_PATH=\"/home/dino/3dvid/sfm/src/sfm/dataset/small_dataset\")
install(TARGETS sfm_node DESTINATION lib/${PROJECT_NAME})



add_executable(train_vocab
    src/train_vocab.cpp
)

target_include_directories(train_vocab PRIVATE
    /usr/local/include  
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(train_vocab
    DBoW2
    ${OpenCV_LIBS}
)

target_compile_definitions(train_vocab PRIVATE DATASET_PATH=\"/home/dino/3dvid/sfm/src/sfm/dataset/small_dataset\")
install(TARGETS train_vocab DESTINATION lib/${PROJECT_NAME})




if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
