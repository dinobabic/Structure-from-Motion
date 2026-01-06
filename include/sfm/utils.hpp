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
#include <tuple>
#include <sophus/se3.hpp>
 
cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(10000);

struct MatchInfo {
    int kp_idx_this;   // keypoint index in this image
    int kp_idx_other;  // keypoint index in the other image
    int other_img_id;  // id of the other image
    int point3D_idx = -1; // idx of 3D point this match triangulates to

    bool operator==(const MatchInfo &other) const {
        return kp_idx_this == other.kp_idx_this &&
            kp_idx_other == other.kp_idx_other &&
            other_img_id == other.other_img_id;
    }
};

struct MatchInfoHash {
    std::size_t operator()(const MatchInfo &m) const {
        return std::hash<int>{}(m.kp_idx_this) ^
               (std::hash<int>{}(m.kp_idx_other) << 1) ^
               (std::hash<int>{}(m.other_img_id) << 2);
    }
};

struct ImageDescription
{
    int img_id;
    std::vector<cv::Point2f> pts;
    std::vector<cv::KeyPoint> kps;
    cv::Mat descriptors;
    std::unordered_set<MatchInfo, MatchInfoHash> matches; 
};

struct Observation
{   
    Observation(int _camera_id, int _point_id, int _keypoint_id, const Eigen::Vector2f &_p) : 
        camera_id(_camera_id),
        point_id(_point_id),
        keypoint_id(_keypoint_id),
        p(_p) {}

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

void detect_features(const cv::Mat &img, ImageDescription &img_description)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);

    img_description.kps = keypoints;
    img_description.descriptors = descriptors;

    for (int i = 0; i < img_description.kps.size(); i++)
        img_description.pts.push_back(img_description.kps[i].pt);

    std::cout << "The number of detected features in image " << img_description.img_id << " is: " << img_description.kps.size() << std::endl;
}

void match_features(ImageDescription &img_description1, ImageDescription &img_description2)
{
    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf.knnMatch(img_description1.descriptors, img_description2.descriptors, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() == 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }  

    for (const auto &match : good_matches)
    {
        img_description1.matches.insert({match.queryIdx, match.trainIdx, img_description2.img_id});
        img_description2.matches.insert({match.trainIdx, match.queryIdx, img_description1.img_id});
    }
}

void match_features_pnp(
    const ImageDescription &img_description1,   // has 3D points
    ImageDescription &img_description2          // new image
)
{
    cv::Mat query_descriptors;
    std::vector<int> query_kp_indices;
    std::vector<int> query_point3D_indices;

    for (const auto &m : img_description1.matches)
    {
        query_descriptors.push_back(img_description1.descriptors.row(m.kp_idx_this));
        query_kp_indices.push_back(m.kp_idx_this);
        query_point3D_indices.push_back(m.point3D_idx);
    }

    std::cout << "Number of points in image " << img_description1.img_id << " that have 3D point is: " << query_kp_indices.size() << std::endl;

    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf.knnMatch(query_descriptors, img_description2.descriptors, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() < 2)
            continue;

        const auto &m1 = knn_matches[i][0];
        const auto &m2 = knn_matches[i][1];

        if (m1.distance < ratio_thresh * m2.distance)
        {
            int query_row = m1.queryIdx;
            int train_kp  = m1.trainIdx;

            MatchInfo match;
            match.kp_idx_this  = train_kp;                     
            match.kp_idx_other = query_kp_indices[query_row];  
            match.other_img_id = img_description1.img_id;
            match.point3D_idx  = query_point3D_indices[query_row];

            img_description2.matches.insert(match);
        }
    }

    std::cout << "The number of matches between image " << img_description1.img_id << " and image " << img_description2.img_id << " is: " << img_description2.matches.size() << std::endl;
}

void match_features_triangulation(
    ImageDescription &img_description1,   // has 3D points
    ImageDescription &img_description2    // has 3D points
)
{
    // match all descriptors
    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf.knnMatch(img_description1.descriptors, img_description2.descriptors, knn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    int old_size = img_description2.matches.size();

    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() < 2)
            continue;

        const auto &m1 = knn_matches[i][0];
        const auto &m2 = knn_matches[i][1];

        if (m1.distance < ratio_thresh * m2.distance)
        {
            good_matches.push_back(m1);
            int query_row = m1.queryIdx;
            int train_row = m1.trainIdx;

            if (!img_description1.matches.count({query_row, train_row, img_description2.img_id}))
            {
                img_description1.matches.insert(MatchInfo{query_row, train_row, img_description2.img_id});
                img_description2.matches.insert(MatchInfo{train_row, query_row, img_description1.img_id});
            }
        }
    }

    std::cout << "In triangulation matching, the number of good matches is: " << good_matches.size() << ", while number of accepted matches is: " << img_description2.matches.size() - old_size << std::endl;
}


void detect_and_match(const cv::Mat img1, const cv::Mat img2,
                        ImageDescription &img_description1, ImageDescription &img_description2)
{
    detect_features(img1, img_description1);
    detect_features(img2, img_description2);
    match_features(img_description1, img_description2);
}   

void get_matches_from_descriptions(const ImageDescription &img_description1, const ImageDescription &img_description2, 
                                    std::vector<cv::Point2f>& kps1, std::vector<cv::Point2f>& kps2,
                                    std::vector<int> &kps_idx1, std::vector<int> &kps_idx2)
{
    for (const auto &m : img_description1.matches) {
        if (m.other_img_id == img_description2.img_id) {
            kps1.push_back(img_description1.pts[m.kp_idx_this]);
            kps2.push_back(img_description2.pts[m.kp_idx_other]);
            kps_idx1.push_back(m.kp_idx_this);
            kps_idx2.push_back(m.kp_idx_other);
        }
    }   
}

void filter_matches(ImageDescription &img1, ImageDescription &img2,
                    const std::vector<int> &kps_idx1, const std::vector<int> &kps_idx2,
                    const cv::Mat &mask)
{
    std::unordered_set<MatchInfo, MatchInfoHash> filtered1;
    std::unordered_set<MatchInfo, MatchInfoHash> filtered2;

    for (int i = 0; i < kps_idx1.size(); i++)
    {
        if (mask.at<uchar>(i)) 
        {
            auto it1 = img1.matches.find({kps_idx1[i], kps_idx2[i], img2.img_id});
            if (it1 != img1.matches.end()) filtered1.insert(*it1);

            auto it2 = img2.matches.find({kps_idx2[i], kps_idx1[i], img1.img_id});
            if (it2 != img2.matches.end()) filtered2.insert(*it2);
        }
    }

    img1.matches = std::move(filtered1);
    img2.matches = std::move(filtered2);
}

void visualize_matches(const cv::Mat &img1, const cv::Mat &img2, 
                        const ImageDescription &img_description1, const ImageDescription &img_description2)
{   
    std::vector<cv::Point2f> kps1, kps2;
    std::vector<int> kps_idx1, kps_idx2;
    get_matches_from_descriptions(img_description1, img_description2, kps1, kps2, kps_idx1, kps_idx2);

    std::vector<cv::DMatch> dummy_matches;
    for (size_t i = 0; i < kps_idx1.size(); i++) {
        dummy_matches.emplace_back(kps_idx1[i], kps_idx2[i], 0.0f);
    }

    cv::Mat img_matches;
    cv::drawMatches(img1, img_description1.kps, img2, img_description2.kps, dummy_matches, img_matches);
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

void colorize_points(const cv::Mat img1, const cv::Mat img2,
                    std::vector<Eigen::Vector3f> &colors,
                    int N, // colorize last N point in points3D vector 
                    const std::vector<Eigen::Vector3f> &points3D,
                    const Eigen::Matrix3f &K,
                    const Sophus::SE3f &T1,
                    const Sophus::SE3f &T2)
{
    for (int i = points3D.size()-N; i < points3D.size(); i++)
    {  
        auto point = points3D[i];
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
}

void cv_pose_to_eigen_pose(const cv::Mat &R, const cv::Mat &t,
                          Eigen::Matrix3f &_R, Eigen::Vector3f &_t)
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            _R(r, c) = R.at<float>(r, c);

    _t(0) = t.at<float>(0);
    _t(1) = t.at<float>(1);
    _t(2) = t.at<float>(2);
}

void sophus_pose_to_cv(const Sophus::SE3f &P, cv::Mat &P_cv)
{
    auto R_cv = cv::Mat(3, 3, CV_32F);
    auto t_cv = cv::Mat(3, 1, CV_32F);

    Eigen::Matrix3f R_eigen = P.rotationMatrix();
    Eigen::Vector3f t_eigen = P.translation();

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
            R_cv.at<float>(r, c) = R_eigen(r, c);

        t_cv.at<float>(r, 0) = t_eigen(r);
    }

    cv::hconcat(R_cv, t_cv, P_cv);
}