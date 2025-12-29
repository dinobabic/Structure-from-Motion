#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <DBoW2/DBoW2.h>
#include <unordered_set>
#include <sophus/se3.hpp>

struct Matches
{
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    std::vector<cv::KeyPoint> kps1;
    std::vector<cv::KeyPoint> kps2;
    std::vector<cv::DMatch> matches;
    cv::Mat descriptors1;
    cv::Mat descriptors2;
};

struct Observation
{   
    Observation(int _camera_id, int _point_id, int _keypoint_id, const Eigen::Vector2f &_p)
    {
        camera_id = _camera_id;
        point_id = _point_id;
        keypoint_id = _keypoint_id;
        p = _p;
    }

    int camera_id;
    int point_id;
    int keypoint_id;
    Eigen::Vector2f p;
};

std::vector<std::string> get_image_names(const std::string &dataset_path)
{
    std::vector<std::string> image_names;
    for (const auto &entry : std::filesystem::directory_iterator(dataset_path))
    {
        if (entry.is_regular_file()) 
        {
            auto ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".JPG" || ext == ".PNG") 
            {
                image_names.push_back(entry.path());
            }
        }
    }

    std::sort(image_names.begin(), image_names.end());
    return image_names;
}

std::vector<cv::Mat> load_imgs(const std::vector<std::string> &image_names)
{
    std::vector<cv::Mat> imgs;
    for (const auto &img_name : image_names)
        imgs.push_back(cv::imread(img_name, cv::IMREAD_GRAYSCALE));
    return imgs;
}

std::vector<std::vector<cv::Mat>> get_image_descriptors(const std::vector<cv::Mat> &imgs)
{
    std::vector<std::vector<cv::Mat>> descriptors;
    descriptors.reserve(imgs.size());
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();

    for (const auto &img : imgs) 
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(img, cv::Mat(), keypoints, descriptor);
        
        // descriptor is a matrix of size (N x descriptor_size)
        // DBoW2 wants a vector of descriptors each stored as a separate cv::Mat
        std::vector<cv::Mat> descriptors_img;
        for (size_t i = 0; i < descriptor.rows; i++)
            descriptors_img.push_back(descriptor.row(i));
        descriptors.push_back(descriptors_img);
    }

    return descriptors;
}

Eigen::Matrix3f read_calib_matrix(const std::string &path) 
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cout << "Could not open file " << path << std::endl;
        throw std::invalid_argument("File does not exist.");
    }

    Eigen::Matrix3f K; 
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            file >> K(i,j);

    file.close();
    return K;
}

int retrieve_best_match(OrbDatabase &db, std::vector<cv::Mat> descriptor, const std::unordered_set<int> &visited_frames) 
{
    DBoW2::QueryResults ret;
    db.query(descriptor, ret, db.size());
    for (const auto &res : ret)
    {   
        if (!visited_frames.count(res.Id))
            return res.Id;
    }

    throw std::runtime_error("Failed retrieving the best match.");
}

Matches detect_and_compute(const cv::Mat img1, const cv::Mat img2)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(10000);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const float ratio_thresh = 0.8f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() == 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    Matches kpmatches;
    int idx1 = 0, idx2 = 0;
    for (const auto &match : good_matches)
    {
        kpmatches.pts1.push_back(keypoints1[match.queryIdx].pt);
        kpmatches.pts2.push_back(keypoints2[match.trainIdx].pt);
        kpmatches.kps1.push_back(keypoints1[match.queryIdx]);
        kpmatches.kps2.push_back(keypoints2[match.trainIdx]);
        kpmatches.matches.emplace_back(idx1, idx2, match.distance);

        idx1++;
        idx2++;
    }

    kpmatches.descriptors1 = descriptors1;
    kpmatches.descriptors2 = descriptors2;

    return kpmatches;
}   

void filter_matches(Matches &matches, const cv::Mat &mask)
{
    std::vector<cv::Point2f> filtered_pts1, filtered_pts2;
    std::vector<cv::KeyPoint> filtered_kps1, filtered_kps2;
    cv::Mat filtered_descs1, filtered_descs2;
    std::vector<cv::DMatch> filtered_matches;

    int idx1 = 0, idx2 = 0; 
    for (size_t i = 0; i < matches.pts1.size(); ++i)
    {
        if (mask.at<uchar>(i))
        {
            filtered_pts1.push_back(matches.pts1[i]);
            filtered_pts2.push_back(matches.pts2[i]);
            filtered_kps1.push_back(matches.kps1[i]);
            filtered_kps2.push_back(matches.kps2[i]);

            filtered_descs1.push_back(
                matches.descriptors1.row(matches.matches[i].queryIdx));
            filtered_descs2.push_back(
                matches.descriptors2.row(matches.matches[i].trainIdx));

            filtered_matches.emplace_back(idx1, idx2, matches.matches[i].distance);

            idx1++;
            idx2++;
        }
    }

    matches.pts1 = filtered_pts1;
    matches.pts2 = filtered_pts2;
    matches.kps1 = filtered_kps1;
    matches.kps2 = filtered_kps2;
    matches.descriptors1 = filtered_descs1;
    matches.descriptors2 = filtered_descs2;
    matches.matches = filtered_matches;
}


void visualize_matches(const cv::Mat &img1, const cv::Mat &img2, const Matches &matches)
{
    cv::Mat img_matches;
    cv::drawMatches(img1, matches.kps1, img2, matches.kps2, matches.matches, img_matches);
    cv::resize(img_matches, img_matches, cv::Size(), 0.5, 0.5);
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

Eigen::Vector2f world2img(Eigen::Vector3f p, Sophus::SE3f T, Eigen::Matrix3f K)
{
    Eigen::Vector3f p_cam = T*p;
    Eigen::Vector3f p_normalized_plane = K*p_cam;
    p_normalized_plane /= p_normalized_plane[2];
    return {p_normalized_plane[0], p_normalized_plane[1]};
}

std::vector<Eigen::Vector3f> colorize_points(const cv::Mat img1, const cv::Mat img2, 
                                            std::vector<Eigen::Vector3f> points3D,
                                            Eigen::Matrix3f K,
                                            Sophus::SE3f T1,
                                            Sophus::SE3f T2)
{
    std::vector<Eigen::Vector3f> colors;

    for (const auto &point : points3D)
    {
        auto p1_img = world2img(point, T1, K);
        auto p2_img = world2img(point, T2, K);

        int x1 = std::round(p1_img[0]);
        int y1 = std::round(p1_img[1]);
        int x2 = std::round(p2_img[0]);
        int y2 = std::round(p2_img[1]);

        x1 = std::max(0, std::min(x1, img1.cols-1));
        y1 = std::max(0, std::min(y1, img1.rows-1));
        x2 = std::max(0, std::min(x2, img2.cols-1));
        y2 = std::max(0, std::min(y2, img2.rows-1));

        cv::Vec3b c1 = img1.at<cv::Vec3b>(y1, x1);
        cv::Vec3b c2 = img2.at<cv::Vec3b>(y2, x2);

        Eigen::Vector3f avg_color;
        avg_color[0] = (c1[2] + c2[2]) / 2.0f; 
        avg_color[1] = (c1[1] + c2[1]) / 2.0f; 
        avg_color[2] = (c1[0] + c2[0]) / 2.0f; 

        colors.push_back({c1[2], c1[1], c1[0]});
    }     


    return colors;
}