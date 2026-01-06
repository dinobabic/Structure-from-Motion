#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <DBoW2/DBoW2.h>
#include <sophus/se3.hpp>

#include "sfm/sfm.hpp"
#include "sfm/bundle_adjustment.hpp"

/*
    PUBLISH THIS TRANSFORM from world to map
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 -1.5708 map world
*/

const std::string dataset_path = DATASET_PATH; // DATASET_PATH is defined in CMakeLists.txt

class SfMNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SfMNode() : Node("sfm_node"), 
                sfm(dataset_path),
                img_names(get_image_names(dataset_path)),
                imgs(load_imgs(img_names)),
                K(read_calib_matrix(dataset_path + "/K.txt"))
    {   
        Eigen::Matrix3f R_cv_to_ros;
        R_cv_to_ros <<
            -1,  0,  0,
            0,  -1,  0,
            0,   0,  1;
        Sophus::SE3f pose_cv_to_ros(R_cv_to_ros, Eigen::Vector3f::Zero());

        point_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("points", 10);
        cameras_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("cameras", 10);

        int N = imgs.size();
        Sophus::SE3f T_1W(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()); // pose of the world in the first camera
        
        poses.push_back(T_1W);
        poses_visualization.push_back(pose_cv_to_ros*T_1W);

        // load DBoW2 database
        db = OrbDatabase(dataset_path + "/db.yml.gz");

        // populate the databse with descriptors
        auto img_descriptors = get_image_descriptors(imgs);
        for (size_t i = 0; i < img_descriptors.size(); i++)
            db.add(img_descriptors[i]);

        
        for (int i = 0; i < N-1; i++)
        {
            int best_match_id;
            int query_img_id;
            if (i == 0) 
            {   
                // 2D-2D motion between two first images
                query_img_id = 0;               
                visited_frames.insert(query_img_id);

                // find the most similar image to the query img
                best_match_id = retrieve_best_match(db, img_descriptors[query_img_id], visited_frames);
                std::cout << "The most similar image to image " << img_names[query_img_id] << " is image " << img_names[best_match_id] << std::endl;
                visited_frames.insert(best_match_id);

                // compute the relative motion between two frames using epipolar geometry
                Eigen::Matrix3f R;
                Eigen::Vector3f t;
                ImageDescription img_description1, img_description2;
                img_description1.img_id = query_img_id;
                img_description2.img_id = best_match_id;

                sfm.motion_2D2D(
                    imgs[query_img_id], 
                    imgs[best_match_id], 
                    img_description1,
                    img_description2,
                    points3D, 
                    observations,
                    R, t
                );

                image_descriptions.push_back(img_description1);
                image_descriptions.push_back(img_description2);
                
                Sophus::SE3f T_2W(R, t); // pose of the first camera in the second camera
                poses.push_back(T_2W); // pose of the world in the second camera  
                
                // perform bundle adjustment
                //solve_bundle_adjustment(poses, points3D, observations, K);
                poses_visualization.push_back(pose_cv_to_ros*poses.back());

                colorize_points(
                    cv::imread(img_names[query_img_id], cv::IMREAD_COLOR), 
                    cv::imread(img_names[best_match_id], cv::IMREAD_COLOR), 
                    colors, points3D.size(),
                    points3D,
                    K,
                    poses[0],
                    poses[1]
                );
            }   
            else
            {
                // 3D-2D motion between point cloud and image
                query_img_id = image_descriptions.back().img_id; // query image is the last processed image
                best_match_id = retrieve_best_match(db, img_descriptors[query_img_id], visited_frames);
                std::cout << "The most similar image to image " << img_names[query_img_id] << " is image " << img_names[best_match_id] << std::endl;
                visited_frames.insert(best_match_id);
                
                // compute the relative motion between two cameras using PnP
                Eigen::Matrix3f R;
                Eigen::Vector3f t;
                ImageDescription img_description;
                img_description.img_id = best_match_id;

                sfm.motion_3D2D(
                    imgs[best_match_id],
                    img_description, 
                    image_descriptions.back(), 
                    points3D, 
                    observations,
                    R, t
                );

                image_descriptions.push_back(img_description);
                
                Sophus::SE3f T_NW(R, t); // pose of the world in the n-th camera
                poses.push_back(T_NW); 
                
                int m = points3D.size(); // number of 3D points before triangulating new
                sfm.triangulate_new_matches(image_descriptions[i-1], img_description, poses[i-1], poses[i], points3D, observations);

                std::cout << "Pose of the third camera before ba: \n" << poses.back().matrix3x4() << std::endl;
                
                // perform bundle adjustment
                //solve_bundle_adjustment(poses, points3D, observations, K);
                poses_visualization.push_back(pose_cv_to_ros*poses.back());

                std::cout << "Pose of the third camera after ba: \n" << poses.back().matrix3x4() << std::endl;

                colorize_points(
                    cv::imread(img_names[i-1], cv::IMREAD_COLOR), 
                    cv::imread(img_names[best_match_id], cv::IMREAD_COLOR), 
                    colors, points3D.size() - m,
                    points3D,
                    K,
                    poses[i-1],
                    poses[i]
                );
            }

            std::cout << "The number of points in point cloud: " << points3D.size() << std::endl;
            point_cloud_pub->publish(point_cloud_to_message());
            publish_camera_poses();
        }
    }   

private:
    SfM sfm;
    OrbDatabase db;
    std::unordered_set<int> visited_frames;

    // opencv and ros2 use different convention for coordinate frames - check rotation matrix
    std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> poses; // pose of each camera in the world frame - opencv
    std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> poses_visualization; // pose of each camera in the world frame - ros2

    std::vector<Eigen::Vector3f> points3D; // 3D points expressed in the coordinate system of the first frame
    std::vector<Eigen::Vector3f> colors; // color of each 3D point in cloud
    std::vector<Observation> observations; // for each 3D point, we must know to which camera and what 2D point it projects to
    Eigen::Matrix3f K;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cameras_pub;

    std::vector<std::string> img_names;
    std::vector<cv::Mat> imgs;
    std::vector<ImageDescription> image_descriptions;

    void publish_camera_poses()
    {
        visualization_msgs::msg::MarkerArray camera_array;
        for (size_t i = 0; i < poses.size(); i++)
        {   
            auto pose = poses_visualization[i];

            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = this->get_clock()->now();
            marker.id = i;
            marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = 0.008;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            float d = 0.5f;
            float fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
            float w = cx* d / fx;
            float h = cy* d / fy;

            Eigen::Vector3f origin = pose.translation();
            Eigen::Vector3f tl = pose * Eigen::Vector3f(-w, h, d);
            Eigen::Vector3f tr = pose * Eigen::Vector3f(w, h, d);
            Eigen::Vector3f bl = pose * Eigen::Vector3f(-w, -h, d);
            Eigen::Vector3f br = pose * Eigen::Vector3f(w, -h, d);

            std::vector<Eigen::Vector3f> points = {tl, tr, br, bl};

            for (auto& p : points) {
                geometry_msgs::msg::Point gp;
                gp.x = origin[0]; gp.y = origin[1]; gp.z = origin[2];
                marker.points.push_back(gp);
                gp.x = p[0]; gp.y = p[1]; gp.z = p[2];
                marker.points.push_back(gp);
            }

            for (size_t i = 0; i < 4; i++) {
                geometry_msgs::msg::Point gp1, gp2;
                gp1.x = points[i][0]; gp1.y = points[i][1]; gp1.z = points[i][2];
                gp2.x = points[(i+1)%4][0]; gp2.y = points[(i+1)%4][1]; gp2.z = points[(i+1)%4][2];
                marker.points.push_back(gp1);
                marker.points.push_back(gp2);
            }

            camera_array.markers.push_back(marker);
        }
        cameras_pub->publish(camera_array);
    }

    sensor_msgs::msg::PointCloud2 point_cloud_to_message()
    {
        sensor_msgs::msg::PointCloud2 msg;
        msg.height = 1;
        msg.width = points3D.size();
        msg.is_dense = true;
        msg.is_bigendian = false;
        msg.header.frame_id = "world";
        msg.header.stamp = this->get_clock()->now();

        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2Fields(
            4,                      // number of fields
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "rgb", 1, sensor_msgs::msg::PointField::FLOAT32
        );
        modifier.resize(points3D.size());

        sensor_msgs::PointCloud2Iterator<float> it_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> it_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> it_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> it_r(msg, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> it_g(msg, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> it_b(msg, "b");

        for (int i = 0; i < points3D.size(); i++) 
        {
            auto p = points3D[i];
            auto c = colors[i];

            *it_x = p[0];
            *it_y = p[1];
            *it_z = p[2];

            *it_r = static_cast<uint8_t>(std::round(c[0]));
            *it_g = static_cast<uint8_t>(std::round(c[1]));
            *it_b = static_cast<uint8_t>(std::round(c[2]));

            ++it_x; ++it_y; ++it_z;
            ++it_r; ++it_g; ++it_b;
        }

        return msg;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SfMNode>());
    rclcpp::shutdown();
    return 0;
}

