#pragma once
#include <eigen3/Eigen/Core>
#include <vector>
#include <string>

#include "sfm/utils.hpp"

class SfM 
{
public: 
    SfM(const std::string &dataset_path) : _dataset_path(dataset_path), K_cv(3, 3, CV_32F)
    {
        // read calibration matrix - assumes known calibration matrix and constant for all images
        K = read_calib_matrix(dataset_path + "/K.txt");

        // fill in K_cv with calibration matrix
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                K_cv.at<float>(r, c) = K(r, c);
    }

    void motion_2D2D(
        int img1_id,
        int img2_id,
        const cv::Mat &img1,
        const cv::Mat &img2,
        std::vector<Eigen::Vector3f> &_points3D,
        std::vector<Observation> &_observations,
        Eigen::Matrix3f &_R,
        Eigen::Vector3f &_t)
    {
        Matches matches = detect_and_compute(img1, img2);
        
        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(
            matches.pts1, matches.pts2, K_cv,
            cv::RANSAC, 0.999, 5.0, inlier_mask);

        filter_matches(matches, inlier_mask);

        cv::Mat R, t;
        inlier_mask = cv::Mat();
        cv::recoverPose(E, matches.pts1, matches.pts2, K_cv,  R, t, inlier_mask);
        R.convertTo(R, CV_32F);
        t.convertTo(t, CV_32F);
        
        filter_matches(matches, inlier_mask);

        std::vector<cv::Point2f> pts1_norm, pts2_norm;
        cv::undistortPoints(matches.pts1, pts1_norm, K_cv, cv::Mat());
        cv::undistortPoints(matches.pts2, pts2_norm, K_cv, cv::Mat());

        cv::Mat P1 = cv::Mat::eye(3,4,CV_32F);
        cv::Mat P2;
        cv::hconcat(R, t, P2);

        cv::Mat points4D;
        cv::triangulatePoints(P1, P2, pts1_norm, pts2_norm, points4D);

        for (size_t i = 0; i < points4D.cols; i++)
        {
            cv::Mat x = points4D.col(i);
            x /= x.at<float>(3);

            if (x.at<float>(2) <= 0) continue;

            _points3D.emplace_back(
                x.at<float>(0),
                x.at<float>(1),
                x.at<float>(2)
            );

            int kp1_id = matches.matches[i].queryIdx;
            int kp2_id = matches.matches[i].trainIdx;

            // observation in the first and second camera
            _observations.push_back(Observation(img1_id, _points3D.size()-1, kp1_id, Eigen::Vector2f(matches.pts1[i].x, matches.pts1[i].y)));
            _observations.push_back(Observation(img2_id, _points3D.size()-1, kp2_id, Eigen::Vector2f(matches.pts2[i].x, matches.pts2[i].y)));
        }

        // fill in Eigen rotation matrix
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                _R(r, c) = R.at<float>(r, c);

        // fill in translation vector
        _t(0) = t.at<float>(0);
        _t(1) = t.at<float>(1);
        _t(2) = t.at<float>(2);

    }

    void motion_3D2D(
        int img1_id,
        int img2_id,
        const cv::Mat &img1,
        const cv::Mat &img2,
        std::vector<Eigen::Vector3f> &_points3D,
        std::vector<Observation> &_observations,
        Eigen::Matrix3f &_R,
        Eigen::Vector3f &_t)
    {
        // for each keypoint in the second image (for which we know 2D-3D correspondance) check if they can be matched with keypoints in the third image
        std::unordered_map<int, int> kp2point;
        for (const auto &obs : _observations)
        {
            if (obs.camera_id == img1_id)
                kp2point[obs.keypoint_id] = obs.point_id;
        }

        Matches matches = detect_and_compute(img1, img2);
        std::vector<cv::Point3f> pts3D;
        std::vector<cv::Point2f> pts2D;
        std::vector<int> pts3Did;
        std::vector<int> pts2Did;

        for (const auto &m : matches.matches)
        {
            int kp1_id = m.queryIdx; // id of of a keypoint in query img
            int kp2_id = m.trainIdx; // id of of a keypoint in best match img
        
            if (kp2point.count(kp1_id)) // if this keypoint in the query img has corresponding 3D point
            {
                int point3D_id = kp2point[kp1_id];
                auto point3D = _points3D[point3D_id];
                auto point2D = matches.pts2[kp2_id];

                pts3D.push_back(cv::Point3f(point3D[0], point3D[1], point3D[2]));
                pts2D.push_back(point2D);
                pts3Did.push_back(point3D_id);
                pts2Did.push_back(kp2_id);
                //_observations.push_back(Observation(img2_id, point3D_id, kp2_id, Eigen::Vector2f(point2D.x, point2D.y)));
            }
        }

        cv::Mat R, R_vec, t, mask;
        cv::solvePnPRansac(
            pts3D,
            pts2D,
            K_cv,
            cv::Mat(),
            R_vec,
            t,
            false,
            100,
            8.0,
            0.99, 
            mask,
            cv::SOLVEPNP_ITERATIVE
        );

        // filter outliers
        for (int i = 0; i < pts3D.size(); i++)
        {
            if (mask.at<uchar>(i))
                _observations.push_back(Observation(img2_id, pts3Did[i], pts2Did[i], Eigen::Vector2f(pts2D[i].x, pts2D[i].y)));
        }

        cv::Rodrigues(R_vec, R);
    
        // fill in Eigen rotation matrix
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                _R(r, c) = R.at<float>(r, c);

        // fill in translation vector
        _t(0) = t.at<float>(0);
        _t(1) = t.at<float>(1);
        _t(2) = t.at<float>(2);
    }   


private:
    std::string _dataset_path;
    Eigen::Matrix3f K;
    cv::Mat K_cv;

    void recover_pose( const std::vector<cv::Point2f>& pts1, 
                              const std::vector<cv::Point2f>& pts2,
                              const cv::Mat& P1,
                              const cv::Mat& P2,
                              std::vector<cv::Point3f>& points3D,
                              cv::Mat& mask)
    {
        
        cv::Mat points4D; // triangulated 3D points - homogeneous representation
        cv::triangulatePoints(P1, P2, pts1, pts2, points4D);
        mask = cv::Mat::zeros(points4D.cols, 1, CV_8U);
        for (size_t i = 0; i < points4D.cols; i++)
        {
            cv::Mat point4D = points4D.col(i);
            point4D /= point4D.at<float>(3);
            cv::Point3f point3D(point4D.at<float>(0), point4D.at<float>(1), point4D.at<float>(2)); // point 3D in the first camera
            cv::Mat point3D_hom = (cv::Mat_<float>(4,1) << point3D.x, point3D.y, point3D.z, 1.0f);
            cv::Mat point3D_second_camera = P2 * point3D_hom; // point 3D in the second camera
            if (point3D.z > 0 && point3D_second_camera.at<float>(2) > 0) // valid points are those that lie in front of both cameras - depth is positive
            {
                points3D.push_back(point3D);
                mask.at<uchar>(i) = 1;
            }
        }

    }
};