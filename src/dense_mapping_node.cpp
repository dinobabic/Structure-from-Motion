#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <fstream>
#include <boost/timer/timer.hpp>
#include <string>
#include <vector>

#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


const int border = 20;         
const int width = 640;          
const int height = 480;         
const double fx = 481.2f;       
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int window_size = 3;    
const int window_area = (2 * window_size + 1) * (2 * window_size + 1); 

bool read_dataset_files(
    const std::string &path,
    std::vector<std::string> &color_image_files,
    std::vector<Sophus::SE3d> &poses
)
{
    std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof())
    {
        std::string image;
        fin >> image;
        double data[7];
        for (double &d : data) fin >> d;
        
        color_image_files.push_back(path + "/images/" +image);
        poses.push_back(
            Sophus::SE3d(Eigen::Quaterniond(data[6], data[3], data[4], data[5]), Eigen::Vector3d(data[0], data[1], data[2]))
        );

        if (!fin.good()) break;
    }

    return true;
}

void plot_depth(const cv::Mat &depth)
{
    cv::imshow("depth", depth);
    cv::waitKey(1);
}

inline Eigen::Vector2d cam2pix(const Eigen::Vector3d &point_camera)
{
    return Eigen::Vector2d(
        point_camera(0,0)*fx / point_camera(2,0) + cx,
        point_camera(1,0)*fy / point_camera(2,0) + cy
    );
}

inline Eigen::Vector3d pix2cam(const Eigen::Vector2d &point_pixel)
{
    return Eigen::Vector3d(
        (point_pixel(0,0)-cx) / fx,
        (point_pixel(1,0)-cy) / fy,
        1
    );
}

void show_epipolar_match(const cv::Mat& ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_curr)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2d(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2d(px_curr(0,0), px_curr(1,0)), 5, cv::Scalar(0, 0, 250), 2);

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}

void show_epipolar_line(const cv::Mat& ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref, const Eigen::Vector2d &px_min_curr, const Eigen::Vector2d &px_max_curr)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2d(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2d(px_min_curr(0,0), px_min_curr(1,0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2d(px_max_curr(0,0), px_max_curr(1,0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::line(curr_show, cv::Point2d(px_min_curr(0,0), px_min_curr(1,0)), cv::Point2d(px_max_curr(0,0), px_max_curr(1,0)), cv::Scalar(0, 0, 250), 2);

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);   
}

class DenseMappingNode : public rclcpp::Node
{
public:
    DenseMappingNode() : Node("dense_mapping_node")
    {
        std::vector<std::string> color_image_files;
        std::vector<Sophus::SE3d> poses_TWC;
        bool ret = read_dataset_files("/home/dino/3dvid/sfm/src/sfm/dataset/large_dataset", color_image_files, poses_TWC);
        if (!ret)
        {
            std::cout << "Reading images failed!" << std::endl;
        }

        std::cout << "Read " << color_image_files.size() << " images." << std::endl;    

        cv::Mat ref = cv::imread(color_image_files[0], cv::IMREAD_GRAYSCALE);
        Sophus::SE3d pose_ref_TWC = poses_TWC[0];

        cv::Mat curr = cv::imread(color_image_files[1], cv::IMREAD_GRAYSCALE);
        Sophus::SE3d pose_curr_TWC = poses_TWC[1];
        Sophus::SE3d pose_TCR = pose_curr_TWC.inverse() * pose_ref_TWC; // transformation from ref camera frame to curr camera frame

        for (int x = border; x < width-border; x++)
        {
            for (int y = border; y < height-border; y++)
            {
                Eigen::Vector2d pix_ref(x, y);
                Eigen::Vector2d pix_curr;

                Eigen::Vector3d ray_ref = pix2cam(pix_ref); // ray in the reference camera through the pix_ref
                double d_min = 0.1, d_max = 10, d_step = 0.01;
                
                Eigen::Vector3d point_min_depth_ref = ray_ref*d_min;
                Eigen::Vector3d point_max_depth_ref = ray_ref*d_max;

                // transform these two points to the current camera frame
                Eigen::Vector3d point_min_depth_curr = pose_TCR * point_min_depth_ref;
                Eigen::Vector3d point_max_depth_curr = pose_TCR * point_max_depth_ref;

                // project points in the curr frame to the image plane
                Eigen::Vector2d pix_min_depth_curr = cam2pix(point_min_depth_curr);
                Eigen::Vector2d pix_max_depth_curr = cam2pix(point_max_depth_curr);
                
                // visualize the epipolar line
                show_epipolar_line(ref, curr, pix_ref, pix_min_depth_curr, pix_max_depth_curr);
            }
        }



    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DenseMappingNode>());
    rclcpp::shutdown();
    return 0;
}