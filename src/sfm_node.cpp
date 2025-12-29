#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <DBoW2/DBoW2.h>
#include <sophus/se3.hpp>
#include <unordered_map.hpp>

#include "sfm/sfm.hpp"
#include "sfm/bundle_adjustment.hpp"

const std::string dataset_path = DATASET_PATH; // DATASET_PATH is defined in CMakeLists.txt

class SfMNode : public rclcpp::Node {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SfMNode() : Node("sfm_node"), 
                sfm(dataset_path),
                img_names(get_image_names(dataset_path)),
                imgs(load_imgs(img_names)),
                img_descriptors(get_image_descriptors(imgs)),
                K(read_calib_matrix(dataset_path + "/K.txt"))
    {   
        point_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("points", 10);
        cameras_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("cameras", 10);

        int N = imgs.size();

        // load DBoW2 database
        db = OrbDatabase(dataset_path + "/db.yml.gz");

        // populate the databse with descriptors
        for (size_t i = 0; i < img_descriptors.size(); i++)
            db.add(img_descriptors[i]);

        
        for (int i = 0; i < N; i++)
        {
            int best_match_id;
            int query_img_id;
            if (i == 0) 
            {   
                // 2D-2D motion between two first images
                query_img_id = 0;               
                visited_frames.insert(query_img_id);
                processed_imgs.push_back(query_img_id);

                // find the most similar image to the query img
                best_match_id = retrieve_best_match(db, img_descriptors[query_img_id], visited_frames);
                std::cout << "The most similar image to image " << img_names[query_img_id] << " is image " << img_names[best_match_id] << std::endl;
                visited_frames.insert(best_match_id);
                processed_imgs.push_back(best_match_id);

                // compute the relative motion between two frames using epipolar geometry
                Eigen::Matrix3f R;
                Eigen::Vector3f t;
                sfm.motion_2D2D(
                    query_img_id,
                    best_match_id,
                    imgs[query_img_id], 
                    imgs[best_match_id], 
                    points3D, 
                    observations,
                    R, t
                );

                Sophus::SE3f T_1W(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()); // pose of the world in the first camera
                Sophus::SE3f T_21(R, t); // pose of the first camera in the second camera

                poses.push_back(T_1W);
                poses.push_back(T_21 * T_1W); // pose of the world in the second camera  
                
                // perform bundle adjustment
                solve_bundle_adjustment(poses, points3D, observations, K);
            }   
            else
            {
                // 3D-2D motion between point cloud and image
                query_img_id = processed_imgs.back(); // query image is the last processed image
                best_match_id = retrieve_best_match(db, img_descriptors[query_img_id], visited_frames);
                std::cout << "The most similar image to image " << img_names[query_img_id] << " is image " << img_names[best_match_id] << std::endl;
                visited_frames.insert(best_match_id);
                processed_imgs.push_back(best_match_id);
                
                // compute the relative motion between two cameras using PnP
                Eigen::Matrix3f R;
                Eigen::Vector3f t;
                sfm.motion_3D2D(
                    query_img_id,
                    best_match_id,
                    imgs[query_img_id], 
                    imgs[best_match_id], 
                    points3D, 
                    observations,
                    R, t
                );

                break;
            }
            
            auto colors = colorize_points(cv::imread(img_names[0], cv::IMREAD_COLOR), 
                                    cv::imread(img_names[best_match_id], cv::IMREAD_COLOR), 
                                    points3D,
                                    K,
                                    poses[0],
                                    poses[1]);
            point_cloud_pub->publish(point_cloud_to_message(points3D, colors));
            publish_camera_poses();
        }
    }   

private:
    SfM sfm;
    OrbDatabase db;
    std::unordered_set<int> visited_frames;
    std::vector<int> processed_imgs; // the order in which images have been processed
    std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> poses; // pose of each camera in the world frame
    std::vector<Eigen::Vector3f> points3D; // 3D points expressed in the coordinate system of the first frame
    std::vector<Observation> observations; // for each 3D point, we must know to which camera and what 2D point it projects to
    Eigen::Matrix3f K;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cameras_pub;

    std::vector<std::string> img_names;
    std::vector<cv::Mat> imgs;
    std::vector<std::vector<cv::Mat>> img_descriptors;

    void publish_camera_poses()
    {
        visualization_msgs::msg::MarkerArray camera_array;
        for (size_t i = 0; i < poses.size(); i++)
        {   
            auto pose = poses[i];

            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = this->get_clock()->now();
            marker.ns = "cameras";
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

    sensor_msgs::msg::PointCloud2 point_cloud_to_message(const std::vector<Eigen::Vector3f> &points3D,
                                                        const std::vector<Eigen::Vector3f> &colors)
    {
        sensor_msgs::msg::PointCloud2 msg;
        msg.height = 1;
        msg.width = points3D.size();
        msg.is_dense = true;
        msg.is_bigendian = false;
        msg.header.frame_id = "map";
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

